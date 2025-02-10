from transformers import PreTrainedTokenizer
from datasets import Dataset
from dataclasses import dataclass
import copy
import torch
import hashlib

from config import DatasetEnum, TrainStageEnum, DPOMethodEnum
from tools import tools_json_load, tools_log_on_rank

@dataclass
class DPORefModelOuts:
    uuid: str
    selected: torch.Tensor
    rejected: torch.Tensor

    def __init__(self, uuid: str, selected: torch.Tensor, rejected: torch.Tensor):
        self.uuid = uuid
        self.selected = selected.to('cpu')
        self.rejected = rejected.to('cpu')

    def __hash__(self) -> int:
        return int(hashlib.sha256(self.uuid.encode()).hexdigest(), 16)

    def to_dict(self) -> dict:
        return {'selected': self.selected, 'rejected': self.rejected}

class PaddingCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer, mode: TrainStageEnum, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mode = mode
    
    def padding_sft(self, batch: dict[list]) -> dict[list]:
        """return tensor inputs"""

        tensor_keys = ['input_ids', 'attention_mask', 'labels']

        for key in tensor_keys:
            assert key in batch, f"{key} not in batch = {batch.keys()}"
        
        temp = self.tokenizer.pad(
            {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']},
            padding=True, return_tensors='pt', return_attention_mask=True
        )
        maxlen = temp['input_ids'].shape[1]
        label_pads = [[self.tokenizer.pad_token_id] * (maxlen - len(label)) for label in batch['labels']]

        if self.tokenizer.padding_side == 'left':
            temp['labels'] = torch.tensor([label_pad + label for label_pad, label in zip(label_pads, batch['labels'])])
        else:
            temp['labels'] = torch.tensor([label + label_pad for label_pad, label in zip(label_pads, batch['labels'])])
        
        return temp

    def padding_dpo(self, batch: dict[list]) -> list[dict]:
        selected_rejected_merge = {}

        for k in batch['selected'][0].keys():
            selected_rejected_merge[k] = [item[k] for item in batch['selected']] + [item[k] for item in batch['rejected']]

        return self.padding_sft(selected_rejected_merge)
    
    def padding_pretrain(self, examples: list[dict]) -> list[dict]:
        pass

    def __call__(self, features: list[dict]) -> dict[str, list]:
        batch: dict[list] = {k: [item[k] for item in features] for k in features[0].keys()}

        if self.mode == TrainStageEnum.sft:
            tensor_inputs = self.padding_sft(batch)
            tensor_keys = tensor_inputs.keys()
        
        elif self.mode == TrainStageEnum.dpo:
            tensor_inputs = self.padding_dpo(batch)
            tensor_keys = []
        
        else:
            raise NotImplementedError(f"padding: mode = {self.mode} not implemented")

        return {'tensor_inputs': tensor_inputs, **{k: batch[k] for k in batch.keys() if k not in tensor_keys}}

def check_sanity(examples: list[list[dict]]):
    for i in range(len(examples)):
        assert len(examples[i]) == 2, "prefix must be a list of 2 items"
        assert examples[i][0]['role'] == 'user' and examples[i][1]['role'] == 'assistant', "prefix must be a list of user and assistant messages"

def convert_data_to_sft(data: dict[dict] | dict[list]) -> list[dict]:
    values = list(data.values())
    if isinstance(values[0], list):
        return data
    elif isinstance(values[0], dict):
        # dpo data
        return {uuid: item['selected'] for uuid, item in data.items()}
    else:
        raise NotImplementedError(f"values[0] = {values[0]}")

def preprocess_dataset(data_name: DatasetEnum, stage: TrainStageEnum, tokenizer: PreTrainedTokenizer, max_length: int, dpo_consistency_strategy: None | DPOMethodEnum) -> Dataset:

    def sft_pre(data: list[list[dict]], uuids: list[str]) -> tuple[list[dict], list[str]]:
        """
        some data may be skipped, thus return the correct corresponding uuids for processed data
        """
        # remove the assistant to get the user input
        prefix = copy.deepcopy(data)
        for i in range(len(prefix)): del prefix[i][1]
            
        prefix = tokenizer.apply_chat_template(prefix, padding=False, truncation=False)
        prefix_length = [len(p) for p in prefix]

        inputs = tokenizer.apply_chat_template(data, padding=False, truncation=False, return_dict=True)
        inputs = [{k: v[i] for k, v in inputs.items()} for i in range(len(data))]

        results = []
        new_uuids = []

        for i, (l, item, ori_item, uu) in enumerate(zip(prefix_length, inputs, data, uuids)):
            if len(item['input_ids']) > max_length or len(item['input_ids']) < 5:
                continue
            
            label = copy.deepcopy(item['input_ids'])
            label = [tokenizer.pad_token_id] * l + label[l:]

            results.append(
                {
                    'input_ids': item['input_ids'],
                    'labels': label,
                    'attention_mask': item['attention_mask'],
                    'content': ori_item
                }
            )
            new_uuids.append(uu)
        
        tools_log_on_rank(f"total = {len(data)}, skip = {len(data) - len(results)}, now = {len(results)} examples")

        return results, new_uuids

    
    if stage == TrainStageEnum.sft:
        data = tools_json_load(data_name.value)
        data = convert_data_to_sft(data)
        keys = list(data.keys())
        data = list(data.values())
        check_sanity(data)
        data, keys = sft_pre(data, keys)
        
    elif stage == TrainStageEnum.dpo:
        assert dpo_consistency_strategy is not None, "dpo_consistency_strategy must be provided for dpo stage"

        data = tools_json_load(data_name.value)
        keys = list(data.keys())

        consistency_avg = [item['consistency'] for item in data.values() if item['consistency'] is not None]
        consistency_avg = sum(consistency_avg) / len(consistency_avg)

        selected_sft, selected_uuids = sft_pre([item['selected'] for item in data.values()], keys)
        rejected_sft, rejected_uuids = sft_pre([item['rejected'] for item in data.values()], keys)
        # some data has been skipped due to overlength in the above sft_pre step
        intersection = set(selected_uuids) & set(rejected_uuids)
        selected_sft = {uu: selected_sft[i] for i, uu in enumerate(selected_uuids) if uu in intersection}
        rejected_sft = {uu: rejected_sft[i] for i, uu in enumerate(rejected_uuids) if uu in intersection}
        keys = list(intersection)

        data = [
            {
                'selected': selected_sft[uu],
                'rejected': rejected_sft[uu],
                'consistency': item if (item := data[uu]['consistency']) is not None else consistency_avg,
                'content': data[uu],
            }
            for uu in keys
        ]

        # using average consistency for all examples
        if dpo_consistency_strategy == DPOMethodEnum.consistency_avg:
            for i in range(len(data)): data[i]['consistency'] = consistency_avg
        else:
            pass

    else:
        raise NotImplementedError(f"stage = {stage} not implemented")

    # assign the uuid to identify the data, especially useful for dpo training
    for k, v in zip(keys, data):
        v['uuid'] = k

    return Dataset.from_list(data)
        
        

if __name__ == '__main__':
    from transformers import AutoTokenizer
    from utils import set_pad_token
    from torch.utils.data import DataLoader

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B-Instruct')
    set_pad_token(tokenizer)

    dataset = preprocess_dataset(DatasetEnum.oasst_dpo_1, TrainStageEnum.dpo, tokenizer, 2048, DPOMethodEnum.consistency_dyn)

    print(dataset[0])

    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=PaddingCollator(tokenizer, TrainStageEnum.dpo))

    for batch in dataloader:
        print(batch.keys())
        print({k: len(v) for k, v in batch.items()})
        print({k: len(v) for k, v in batch['tensor_inputs'].items()})
        break