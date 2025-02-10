from transformers import PreTrainedTokenizer, PreTrainedModel
from tools import tools_log_on_rank, tools_is_device_cpu
from config import CommonConfig, ModelEnum, LoraConfig
import os

def set_pad_token(tokenizer: PreTrainedTokenizer, model: PreTrainedModel | None = None, pad_token='<|pad|>'):
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token=pad_token))


    if model is not None:
        current_size = model.get_input_embeddings().num_embeddings

        if len(tokenizer) > current_size:
            model.resize_token_embeddings(len(tokenizer))
            num_new_tokens = len(tokenizer) - current_size
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
    else:
        tools_log_on_rank("the model is None, skip resizing the token embeddings.", level='warning')


    tokenizer.add_eos_token = True
    tokenizer.padding_side = 'left'

def setup(common: CommonConfig):
    import os
    import torch.distributed as dist

    os.environ['WORLD_SIZE'] = str(common.world_size)
    os.environ['MASTER_ADDR'] = common.master_address
    os.environ['MASTER_PORT'] = str(common.master_port)
    os.environ['LOCAL_RANK'] = str(common.rank)
    os.environ['RANK'] = str(common.rank)
    # os.environ['GLOO_SOCKET_IFNAME'] = 'eth0'

    dist.init_process_group("nccl" if not tools_is_device_cpu(common.device) else 'cpu:gloo', rank=common.rank, world_size=common.world_size)

def prepare_model(device: str, model_name_or_path: ModelEnum, lora: LoraConfig | None, bf16=True, debug=False, random_initialized=False, ckpt_path: str | None = None, tokenizer: PreTrainedTokenizer | None = None):
    """
    when loading lora ckpt, the model should have the same embedding size as the tokenizer
    """

    from transformers import AutoModelForCausalLM, AutoConfig
    from peft import PeftModel
    import torch

    from tools import tools_is_lora_ckpt

    if bf16:
        kwargs = {
            'torch_dtype': torch.bfloat16,
            'attn_implementation': "flash_attention_2",
        }
        if not debug or ckpt_path is not None:
            kwargs['device_map'] = device
    else:
        kwargs = {}
    
    kwargs['trust_remote_code'] = True

    if debug and ckpt_path is None:

        if model_name_or_path == ModelEnum.llama3:
            config = AutoConfig.from_pretrained(model_name_or_path.value, num_hidden_layers=1, hidden_size=4096 // 32, intermediate_size=11008 // 32, num_attention_heads=2, num_key_value_heads=2)

        else:
            raise NotImplementedError(f"debug {model_name_or_path}={model_name_or_path.value} not implemented")
        
        model = AutoModelForCausalLM.from_config(config, **kwargs)

    else:
        if random_initialized:
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(model_name_or_path.value, trust_remote_code=True), **kwargs)
        else:
            if (ckpt_path is not None) and (tools_is_lora_ckpt(ckpt_path) is False):
                model = AutoModelForCausalLM.from_pretrained(ckpt_path, **kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model_name_or_path.value, **kwargs)
            
            if ckpt_path is not None and tools_is_lora_ckpt(ckpt_path):
                assert tokenizer is not None, "tokenizer is required for loading LORA ckpt"
                set_pad_token(tokenizer, model)
                model = PeftModel.from_pretrained(model, ckpt_path)
                model = model.merge_and_unload()

    # lora init (not loading)
    # for training
    if lora and lora.enable:
        from peft import LoraConfig

        lora_config = LoraConfig(init_lora_weights=False, task_type='CAUSAL_LM', r=lora.r, lora_alpha=lora.alpha, lora_dropout=lora.dropout)
        model.add_adapter(lora_config)

        tools_log_on_rank(f"add LORA adapter with {lora}, loaded ckpt_path={ckpt_path}")
    
    return model.to(device)

def dist_broadcast_objects(obj, rank: int, group_gloo, src_rank: int = 0) -> any:
    """broadcast an object from src_rank to the whole group"""
    import torch.distributed as dist
    obj_list = [obj] if rank == src_rank else [None]
    dist.broadcast_object_list(obj_list, src=src_rank, group=group_gloo)
    return obj_list[0]

def dist_sync_objects(obj: list[dict] | dict, group_gloo, rank: int, world_size: int, dedup_key: str | None = None) -> list[dict] | dict | None:
    """
    gather all objects across the group: if rank = 0, return merged objects; otherwise None;
    """

    import torch.distributed as dist
    from functools import reduce
    from datetime import timedelta
    from uuid import uuid4

    dist.monitored_barrier(group_gloo, timeout=timedelta(minutes=10))
    gathers = [None for _ in range(world_size)] if rank <= 0 else None
    dist.gather_object(obj, gathers, dst=0, group=group_gloo)

    def merge(x: list | dict, y: list | dict) -> list | dict:
        if isinstance(x, list) and isinstance(y, list): return x + y
        elif isinstance(x, dict) and isinstance(y, dict):
            x.update(y)
            return x
        else:
            raise ValueError(f"Unsupported type: {type(x)} {type(y)}")
    
    def dedup_func(obj) -> int:
        if isinstance(dedup_key, str) and isinstance(obj, dict):
            return obj[dedup_key]
        elif dedup_key is None:
            # all is unique
            return str(uuid4())
        else:
            return hash(obj)
            # raise RuntimeError(f"do not support {type(obj)} for deduplication")

    if rank <= 0:
        merged = reduce(merge, gathers)

        # no deduplication
        if dedup_key is None:
            pass
        else:
            unique = set()

            if isinstance(merged, list):
                reserved = []
                for item in merged:
                    key = dedup_func(item)
                    if key not in unique:
                        unique.add(key)
                        reserved.append(item)
                
            elif isinstance(merged, dict):
                reserved = {}
                for key, item in merged.items():
                    dkey = dedup_func(item)
                    if dkey not in unique:
                        unique.add(dkey)
                        reserved[key] = item
            else:
                raise RuntimeError(f"do not support {type(merged)} for deduplication")
            
            merged = reserved
    else:
        merged = None
    
    return merged