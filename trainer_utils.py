import torch
import os
import math
from transformers import PreTrainedTokenizer, TrainerCallback
from dataclasses import dataclass
from transformers.trainer_callback import TrainerControl, TrainerState
from transformers.training_args import TrainingArguments
from functools import reduce
from tqdm import tqdm
from typing import Any

from config import TrainConfig
from data import DPORefModelOuts


@dataclass
class TrainerMineArgs:
    group_gloo: Any
    args: TrainConfig
    dpo_ref_results: dict[str, DPORefModelOuts] | None
    
class CustomCallback(TrainerCallback):
    def on_save(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        return super().on_save(args, state, control, **kwargs)
        