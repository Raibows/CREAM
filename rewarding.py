
from tqdm import tqdm
import os

from tools import tools_json_load, tools_json_dump, tools_get_checkpoint_load_path, tools_log_on_rank, tools_set_device_env, tools_get_time
from config import RewardingConfig, parse_args



class PaddingCollator:
    def __init__(self, tokenizer: "PreTrainedTokenizer"):
        self.tokenizer = tokenizer
        self.max_length = 4096
    
    def __call__(self, features: list[dict]) -> dict[str, list]:
        self.tokenizer.padding_side = 'left'

        batch = {k: [item[k] for item in features] for k in features[0].keys()}

        batch['response'] = [
            p + [{'role': 'assistant', 'content': r}]
            for p, r in zip(batch['prompt'], batch['response'])
        ]

        prefix = self.tokenizer.apply_chat_template(batch['prompt'], padding=False, truncation=False)
        prefix_length = [len(p) for p in prefix]


        inputs = self.tokenizer.apply_chat_template(batch['response'], padding=True, return_tensors='pt', max_length=self.max_length, truncation=True, return_dict=True)

        # process labels
        labels = inputs['input_ids'].clone()
        for i in range(len(labels)):
            for j in range(len(labels[i])):
                if labels[i, j] != self.tokenizer.pad_token_id:
                    labels[i, j:j+prefix_length[i]] = self.tokenizer.pad_token_id
                    break
            # labels[i, :prefix_length[i]] = self.tokenizer.pad_token_id
        
        inputs['labels'] = labels
        batch['tensor_inputs'] = inputs

        return batch
        
def load_data(input_file: str, debug: bool) -> list[dict]:
    data = tools_json_load(input_file)
    if debug:  data = dict(list(data.items())[:17])
    data = [
        {'uuid': k, 'response_id': i, 'prompt': v['prompt'], 'response': r}
        for k, v in data.items()
        for i, r in enumerate(v['responses'])
    ]
    
    # sorted by length
    data = list(sorted(data, key=lambda x: len(x['prompt']) + len(x['response']), reverse=True))

    return data

def calculate_rewards(args: RewardingConfig, group_gloo, data_loader, model, tokenizer: "PreTrainedTokenizer", desc: str, normalized_by_length: bool) -> list[dict]:
    """
    for no ref method, it is recommened to set normalized_by_length to True to calculate the score
    """

    import torch
    from utils import dist_sync_objects

    model = model.eval()

    outputs = []

    with tqdm(data_loader, total=len(data_loader), desc=desc, disable=args.common.rank != 0) as pbar:
        for batch in data_loader:

            with torch.no_grad():
                logits = model(**batch['tensor_inputs'].to(model.device), return_dict=True)['logits'][:, :-1, :]
                labels = batch['tensor_inputs']['labels'][:, 1:]

                per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
                loss_mask = (labels != tokenizer.pad_token_id).float()
                sum_log_probs = (per_token_logps * loss_mask).sum(-1)

                if normalized_by_length:
                    sum_log_probs /= loss_mask.sum(-1)
                
            for i, likelihood in enumerate(sum_log_probs):
                outputs.append(
                    {
                        'uuid': batch['uuid'][i],
                        'response_id': batch['response_id'][i],
                        'prompt': batch['prompt'][i],
                        'response': batch['response'][i],
                        'dedup_key': batch['uuid'][i] + '_' + str(batch['response_id'][i]),
                        'score': likelihood.item(),
                    }
                )
            
            pbar.update(1)

    # sync across all ranks
    outputs = dist_sync_objects(outputs, group_gloo, args.common.rank, args.common.world_size, dedup_key='dedup_key')

    return outputs
            

def main(rank: int, time_based: str, args: RewardingConfig):
    import torch 
    import torch.distributed as dist

    from datasets import Dataset
    from torch.utils.data import DataLoader, DistributedSampler
    from peft import PeftModel, PeftConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

    from utils import prepare_model, set_pad_token, setup

    # init dist
    args.common.rank = rank
    setup(args.common)
    group_gloo = dist.new_group(backend="gloo")

    # load the data
    data = load_data(args.input_file, args.common.debug)
    data = Dataset.from_list(data)
    tokenizer = AutoTokenizer.from_pretrained(args.model.value)
    set_pad_token(tokenizer)
    
    data_loader = DataLoader(
        data,
        batch_size=args.batch_size,
        collate_fn=PaddingCollator(tokenizer),
        sampler=DistributedSampler(data, args.common.world_size, args.common.rank, shuffle=False,)
    )
    data_loader.sampler.set_epoch(0)

    # rewarding w/ policy model
    model = prepare_model(args.common.rank, args.model, None, args.common.bf16, args.common.debug, ckpt_path=tools_get_checkpoint_load_path(args.checkpoint), tokenizer=tokenizer)
    set_pad_token(tokenizer, model)

    results_policy = calculate_rewards(args, group_gloo, data_loader, model, tokenizer, desc='rewarding policy 1/1' if not args.enable_ref_model else 'rewarding policy 1/2', normalized_by_length=not args.enable_ref_model)

    if args.enable_ref_model:
        # dpo rewards
        del model
        torch.cuda.empty_cache()
        model = prepare_model(args.common.rank, args.model, None, args.common.bf16, args.common.debug, ckpt_path=tools_get_checkpoint_load_path(args.ref_model_ckpt), tokenizer=tokenizer)
        set_pad_token(tokenizer, model)

        results_ref = calculate_rewards(args, group_gloo, data_loader, model, tokenizer, desc='rewarding ref 2/2', normalized_by_length=False)

        if rank == 0:
            results_policy = {
                item['dedup_key']: item for item in results_policy
            }

            # calculate dpo rewards
            for item in results_ref:
                key = item['dedup_key']
                results_policy[key]['ref_score'] = item['score']
                results_policy[key]['reward'] = results_policy[key]['score'] - item['score']
            results_policy = list(results_policy.values())

    else:
        # only likelihood
        if rank == 0:
            for i in range(len(results_policy)):
                results_policy[i]['reward'] = results_policy[i]['score']
                results_policy[i]['ref_score'] = 0
    
    # formalize results
    if rank == 0:
        results = {}
        max_response_num = max(int(item['response_id']) for item in results_policy) + 1
        for item in results_policy:
            if item['uuid'] not in results:
                results[item['uuid']] = {
                    'prompt': item['prompt'],
                    'response': [None] * max_response_num,
                    'score': [None] * max_response_num,
                    'ref_score': [None] * max_response_num,
                    'reward': [None] * max_response_num,
                }

            for key in ['response', 'score', 'ref_score', 'reward']:
                results[item['uuid']][key][item['response_id']] = item[key]

        # save results
        output_path = args.input_file.removesuffix('.json')
        suffix = f"{args.model.name}-policy_{args.checkpoint}-ref_{args.ref_model_ckpt}"
        if args.common.debug: suffix = f"{suffix}-debug"
        final_path = f"{output_path}.rewarding.{suffix}.json"
        if os.path.exists(final_path):
            final_path = f"{output_path}.rewarding.{suffix}.{time_based}.json"
        
        tools_json_dump(results, final_path)
        tools_log_on_rank(args)
        tools_log_on_rank(f"rewarding results are saved to={final_path}")

        



if __name__ == '__main__':
    args: RewardingConfig = parse_args(RewardingConfig, pass_in=[])
    tools_set_device_env(args.common.device)
    time_based = tools_get_time()
    
    tools_log_on_rank(time_based, args)

    import torch.multiprocessing as mp
    mp.spawn(main, args=(time_based, args), nprocs=args.common.world_size, join=True)


