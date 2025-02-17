import os

from simple_parsing import ArgumentParser, choice, Serializable
from dataclasses import dataclass
from enum import Enum

from tools import tools_get_random_available_port, tools_get_simple_dataset_name, tools_log_on_rank, tools_get_model_name, tools_is_device_cpu, tools_assert_ckpt_name_valid

protected_keys = set(['load_args_path'])

def get_hf_training_args(train_config: "TrainConfig", common_config: "CommonConfig", do_eval: bool = True) -> "TrainingArguments":
    # https://huggingface.co/docs/transformers/v4.40.0/en/main_classes/trainer#transformers.TrainingArguments

    from transformers import TrainingArguments

    os.environ['WANDB_MODE'] = 'offline'
    os.environ['WANDB_PROJECT'] = common_config.wandb_project_name
    os.environ['WANDB_ENTITY'] = common_config.wandb_entity_name
    os.environ['WANDB_NOTES'] = common_config.run_name

    hfargs = TrainingArguments(
        output_dir=common_config.output_dir,
        do_train=True,
        do_eval=do_eval,
        do_predict=False,
        report_to=['wandb'],
        logging_steps=1 if common_config.debug else 32,
        ddp_find_unused_parameters=False,
        num_train_epochs=train_config.epoch,
        per_device_train_batch_size=train_config.train_bsz,
        per_device_eval_batch_size=train_config.eval_bsz,
        learning_rate=train_config.lr,
        warmup_ratio=train_config.warmup_step_ratio,
        bf16=common_config.bf16,
        deepspeed=train_config.deepspeed,
        load_best_model_at_end=False,
        resume_from_checkpoint=None,
        overwrite_output_dir=True,
        use_cpu=common_config.device == 'cpu',
        weight_decay=train_config.weight_decay,
        max_grad_norm=1.0,
        run_name=common_config.output_dir.removeprefix('outputs/'),
        save_only_model=True,
        evaluation_strategy='steps' if do_eval else 'no',
        save_steps=train_config.save_eval_step_ratio,
        eval_steps=train_config.save_eval_step_ratio,
        lr_scheduler_type='cosine',
        remove_unused_columns=False,
        logging_first_step=True,
        hub_token=os.getenv('HF_HUB_TOKEN', None),
        gradient_checkpointing=train_config.grad_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    return hfargs


class InitConfig(Serializable):
    def post_check(self):
        for key, value in self.__dict__.items():
            if isinstance(value, str) and value.lower() == 'none':
                self.__dict__[key] = None
            elif isinstance(value, InitConfig):
                value.post_check()
    
    def restore_from_json(self, path: str, drop_extra_fields: bool = True):
        from tools import tools_json_load
        incoming_config: dict = tools_json_load(path)

        def override(config, incoming):
            for key, value in incoming.items():
                if not hasattr(config, key):
                    if drop_extra_fields:
                        tools_log_on_rank(f"key={key}, value={value} from override_args_path is not in current config, thus being ignored", level='warning')
                    else:
                        raise ValueError(f"key={key} is not in current config")
                elif key in protected_keys:
                    tools_log_on_rank(f"key={key} is protected, will not be replaced by value={value} from override_args_path", level='warning')
                
                # is a dict class
                elif isinstance(value, dict):
                    override(config.__dict__[key], value)
                    # should re-post-init
                    config.__dict__[key].post_check()
                    
                else:
                    config.__dict__[key] = value
        
        override(self, incoming_config)

    def recursive_find_key(self, key: str) -> tuple:
        def find_key(config, key):
            for k, v in config.__dict__.items():
                if k == key:
                    return v, True
                elif hasattr(v, '__dict__'):
                    res, flag = find_key(v, key)
                    if flag: return res, flag

            return "", False
        return find_key(self, key)

@dataclass
class CommonConfig(InitConfig):
    debug: bool = False
    device: str = 'cpu'
    world_size: int = 0
    rank: int = 0
    master_address: str = 'localhost'
    master_port: int | None = None
    bf16: bool = True
    wandb_project_name: str = "CREAM"
    wandb_entity_name: str = "your_wandb_entity_name"
    run_name: str | None = None
    output_dir: str | None = None
    # Help: load the command line args from a json file, priority: command line > pass_in > loaded_override_config > default
    load_args_path: str | None = None

    def post_check(self):
        super().post_check()

        if self.device == 'cpu':
            self.world_size = 1
        else:
            self.world_size = len(self.device.split(','))
            if self.world_size > 1:
                self.rank = 'waiting to be assigned'
            else:
                self.rank = 0
        
        if self.master_port is None: self.master_port = tools_get_random_available_port()
        
        if tools_is_device_cpu(self.device) and self.debug:
            tools_log_on_rank("reset bf16 to False since device=cpu and in debug mode", level='warning')
            self.bf16 = False

    def get_device(self) -> int | str:
        """return the virtual rank or cpu"""
        if tools_is_device_cpu(self.device): return 'cpu'
        else: return int(self.rank)
    
    def get_physical_device_id(self) -> int | str:
        """return the physical actual device id, i.e., the nvidia gpu order or cpu"""
        if tools_is_device_cpu(self.device): return 'cpu'
        else: return int(self.device.split(',')[self.get_device()])

    def set_run_name_and_output_dir(self, time_based: str, dataset_config: "DatasetConfig", train_config: "TrainConfig") -> str:
        """
        run_name is the basename of the directory
        output_dir = prefix/run_name
        """
        # output dir
        output_dir = ['outputs']
        if self.debug:
            output_dir.append('debug')
        output_dir.append(train_config.train_stage.name)

        # run name
        if train_config.lora.enable:
            lorainfo = f"lora_{train_config.lora.r}-{train_config.lora.alpha}-{train_config.lora.dropout}"
        else:
            lorainfo = "lora_None"
        
        ckpt = f'ckpt_{train_config.checkpoint}'
        run_name = f"{dataset_config.name.name}-{lorainfo}-{ckpt}"

        # dpo description
        if train_config.train_stage == TrainStageEnum.dpo:
            run_name = f"{run_name}-dpo_{train_config.dpo.method.name}"
        
        # time based 
        run_name = f"{run_name}-{time_based}"

        # format
        output_dir.append(run_name)
        self.output_dir = '/'.join(output_dir)
        self.run_name = run_name

class TrainStageEnum(Enum):
    dpo = "dpo"
    sft = "sft"

class DatasetEnum(Enum):
    oasst_sft = "localdata/oasst1.sft.json"
    all_prompt = "localdata/prompt.json"
    prompt_1 = "localdata/prompt.1.json"
    prompt_2 = "localdata/prompt.2.json"
    prompt_3 = "localdata/prompt.3.json"

    # downstream tasks test set
    arc_easy_test = "localdata/arc-easy/test.json"
    arc_challenge_test = "localdata/arc-challenge/test.json"
    openbookqa_test = "localdata/openbookqa/test.json"
    siqa_test = "localdata/siqa/test.json"
    gsm8k_test = "localdata/gsm8k/test.json"
    # merge 5 tasks
    all_test = "localdata/all_test.json"

    # dpo data
    dpo_1 = "localdata/dpo.1.json"



class ModelEnum(Enum):
    llama3 = "meta-llama/Meta-Llama-3-8B-Instruct"
    llama2 = "meta-llama/Llama-2-7b-chat-hf"

@dataclass
class DatasetConfig(InitConfig):
    name: DatasetEnum = DatasetEnum.oasst_sft
    limit_size: int | None = None
    max_length: int = 2048

    def post_check(self):
        return super().post_check()

    def get_downstream_tasks(self) -> list[DatasetEnum]:
        return [DatasetEnum.arc_easy_test, DatasetEnum.arc_challenge_test, DatasetEnum.openbookqa_test, DatasetEnum.siqa_test, DatasetEnum.gsm8k_test]

    def get_math_tasks(self) -> set[DatasetEnum]:
        return {DatasetEnum.gsm8k_test}
    
    def get_qa_tasks(self) -> set[DatasetEnum]:
        return {DatasetEnum.arc_easy_test, DatasetEnum.arc_challenge_test, DatasetEnum.openbookqa_test, DatasetEnum.siqa_test}

@dataclass
class LoraConfig(InitConfig):
    enable: bool = False
    alpha: int = 64
    r: int = 32
    dropout: float = 0.1

class DPOMethodEnum(Enum):
    original = "dpo_original"
    consistency_avg = "dpo_consistency_avg"
    consistency_dyn = "dpo_consistency_dynamic"

@dataclass
class DPOConfig(InitConfig):
    beta: float = 0.1
    method: DPOMethodEnum = DPOMethodEnum.original

@dataclass
class TrainConfig(InitConfig):
    epoch: int = 1
    train_bsz: int = 2
    eval_bsz: int | None = None
    lr: float = 1e-6
    # the path to deepspeed config json
    deepspeed: str | None = None
    weight_decay: float = 0.01
    # help if none will save every epoch; if >= 1.0, will be save_total times
    save_eval_step_ratio: float | None = None
    warmup_step_ratio: float = 0.1
    # help: use grad_checkpointing can save a lot of memory
    grad_checkpointing: bool = False
    model: ModelEnum = ModelEnum.llama3
    common: CommonConfig = CommonConfig

    # load the previously trained ckpt for further training
    # you have to add it in the configs/checkpoint.json first before use
    checkpoint: str | None = None


    train_stage: TrainStageEnum = TrainStageEnum.sft
    dataset: DatasetConfig = DatasetConfig
    lora: LoraConfig = LoraConfig
    dpo: DPOConfig = DPOConfig
    
    def post_check(self):
        super().post_check()
        if self.eval_bsz is None:
            self.eval_bsz = self.train_bsz * 2

        if self.save_eval_step_ratio is None:
            self.save_eval_step_ratio = self.epoch

        if self.save_eval_step_ratio > 1.0 - 1e-6:
            self.save_eval_step_ratio = int(self.save_eval_step_ratio)
            self.save_eval_step_ratio = (1-1e-6) / self.save_eval_step_ratio
        
        tools_assert_ckpt_name_valid(self.checkpoint)

@dataclass
class GenSamplingConfig(InitConfig):
    gen_n: int = 5
    gen_max_tokens: int = 768
    gen_temperature: float = 0.8
    do_sample: bool = True
    top_p: float = 0.95

    def __repr__(self) -> str:
        if self.do_sample: sample = "Sample"
        else: sample = "Greedy"

        return f"{sample}_n-{self.gen_n}_temp-{self.gen_temperature}_GenMaxT-{self.gen_max_tokens}"
            
    def post_check(self):
        super().post_check()
        if self.gen_temperature < 1e-3:
            tools_log_on_rank(f"temperature={self.gen_temperature} is too small, reset to 0, thus do_sample=False", level='warning')
            self.do_sample = False
            self.gen_temperature = 0

        if self.do_sample is False:
            self.gen_n = 1
            self.gen_temperature = 0
            tools_log_on_rank("do_sample is set to False, thus gen_n=1, gen_temperature=0", level='warning')

class InferModeEnum(Enum):
    evaluation = "evaluation"
    sampling = "sampling"

class SystemPromptEnum(Enum):
    none = "empty"
    task = "task"

@dataclass
class InferVllmConfig(InitConfig):
    common: CommonConfig = CommonConfig
    model: ModelEnum = ModelEnum.llama3
    dataset: DatasetConfig = DatasetConfig
    generation: GenSamplingConfig = GenSamplingConfig
    # load checkpoints
    checkpoint: str | None = None
    tensor_parallel_size: int = 1
    # mode: evaluation or sampling
    mode: InferModeEnum = InferModeEnum.sampling
    # system prompt
    sysprompt: SystemPromptEnum = SystemPromptEnum.none

    def post_check(self):
        super().post_check()
        assert self.common.device != 'cpu', "vllm should not be on cpu"
        assert self.common.bf16, "vllm should use bf16"
        assert self.tensor_parallel_size >= 1 and self.tensor_parallel_size <= self.common.world_size and self.common.world_size % self.tensor_parallel_size == 0, "tensor_parallel_size should be a factor of world_size"
        tools_assert_ckpt_name_valid(self.checkpoint)

        if self.mode == InferModeEnum.evaluation:
            assert self.dataset.name.name.endswith('test'), f"evaluation mode should be on test dataset but got {self.dataset.name.name}"
        elif self.mode == InferModeEnum.sampling:
            assert not self.dataset.name.name.endswith('test'), f"sampling mode should be on prompt dataset but got {self.dataset.name.name}"

        # adjust the generation sampling params for evaluation
        if self.mode == InferModeEnum.evaluation and self.generation.gen_n != 1:
            self.generation.gen_n = 1
            self.generation.do_sample = False
            self.generation.gen_temperature = 0
            tools_log_on_rank("evaluation mode, thus reset gen_n=1, gen_temperature=0, do_sample=False, greedy", level='warning')
        
        if self.mode == InferModeEnum.evaluation:
            self.sysprompt = SystemPromptEnum.task
            tools_log_on_rank("evaluation mode, thus set sysprompt=task", level='warning')

@dataclass
class RewardingConfig(InitConfig):
    common: CommonConfig = CommonConfig
    model: ModelEnum = ModelEnum.llama3
    input_file: str = None
    # if not, calculate the likelihood; otherwise, dpo reward
    enable_ref_model: bool = True
    ref_model_ckpt: str | None = None

    checkpoint: str | None = None
    batch_size: int = 6

    def post_check(self):
        super().post_check()
        tools_assert_ckpt_name_valid(self.checkpoint)
        tools_assert_ckpt_name_valid(self.ref_model_ckpt)
        assert isinstance(self.input_file, str) and os.path.exists(self.input_file), f"input_file={self.input_file} should be a valid file"
        if self.enable_ref_model:
            assert self.checkpoint != self.ref_model_ckpt, f"checkpoint and ref_model_ckpt should be different, but all={self.checkpoint}"


class ConsistencyMethodEnum(Enum):
    kendall = "kendalltau"
    spearman = "spearmanr"
    toporder = "toporder"

@dataclass
class CalConsistencyConfig(InitConfig):
    """file1 will also be used to generate the DPO preference pair data"""
    file1: str = None
    file2: str = None
    method: ConsistencyMethodEnum = ConsistencyMethodEnum.kendall
    output: str | None = None
    
    def post_check(self):
        super().post_check()
        assert self.file1 and os.path.exists(self.file1), f"file1={self.file1} should be a valid file"

        if self.file2 is None:
            self.file2 = self.file1
            tools_log_on_rank(f"file2 is not set, thus file2={self.file1}, NO rewarding supervision", level='warning')



def parse_args(args_config: type[InitConfig], pass_in: list[str] | None = None, 
               enable_restore_from_json: bool = True, allow_unknown_params: bool = False):
    """
    in a jupyter, you may want to enable the allow_unknown_params
    """

    import sys as _sys

    def parse_to_keys_and_values(args: list[str]) -> dict:
        # the key must start_with "--"
        key_value = dict()
        for i, arg in enumerate(args):
            if arg == '--help':
                key_value[arg] = ""

            elif arg.startswith('--'):
                assert arg not in key_value, f"key={arg} should not be duplicated"
                assert i+1 < len(args), f"key={arg} should have a following value, but not found"
                assert not args[i+1].startswith('--'), f"key={arg} should have a following value, but got another key={args[i+1]}"
                key_value[arg] = args[i+1]

            elif i == 0 or i % 2 == 0:
                raise RuntimeError(f"position={i}, value={arg} should be a key. The pass in args must be the strict key-value format.")
                
        return key_value
    
    def restore_keys_values_to_args(key_value: dict) -> list[str]:
        args = []
        for key, value in key_value.items():
            args.extend([key, value])
        return args

    # priority:  command line > pass_in > loaded_override_config > default
    if isinstance(pass_in, list) and len(pass_in) > 0:
        tools_log_on_rank(f"code has pass_in={pass_in}, please make sure you know what you are doing", level='warning')
        pass_in_key_value = parse_to_keys_and_values(pass_in)
        cmd_key_value = parse_to_keys_and_values(_sys.argv[1:])

        # only assign the key-value that not in cmd_key_value
        for key, value in pass_in_key_value.items():
            if key not in cmd_key_value:
                cmd_key_value[key] = value

        pass_in = restore_keys_values_to_args(cmd_key_value)
    else:
        pass_in = _sys.argv[1:]
    
    # default config
    default_config: InitConfig = args_config()
    
    # try to find "--load_args_path"
    args_load_key = "--load_args_path"
    pass_in_key_value = parse_to_keys_and_values(pass_in)
    args_load_path = pass_in_key_value.get(args_load_key, None)
    if args_load_path is not None:
        if os.path.exists(args_load_path) and enable_restore_from_json:
            default_config = default_config.load(args_load_path, drop_extra_fields=True)
        else:
            tools_log_on_rank(f"load_args_path={args_load_path} is passed in, but not restored, since this function is being disabled={enable_restore_from_json} or the load_args_path not existed", level='warning')

    parser = ArgumentParser()
    parser.add_arguments(default_config, "config")

    if allow_unknown_params:
        args, unknown = parser.parse_known_args(pass_in)
        if len(unknown) > 0:
            tools_log_on_rank(f"unknown pass_in params\n{unknown}", level='warning')
    else:
        args = parser.parse_args(pass_in)

    args: InitConfig  = args.config
    args.post_check()

    return args

if __name__ == "__main__":
    args = parse_args(TrainConfig)
    print(args)
    print(args.to_dict())