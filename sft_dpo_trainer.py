import torch
from torch import nn
from transformers import Trainer, PreTrainedModel
from transformers.trainer_callback import ProgressCallback
from typing import Dict, Optional, Union, Any
from tqdm import tqdm
import os


from config import TrainConfig, TrainStageEnum, DPOMethodEnum
from tools import tools_log_on_rank
from trainer_utils import TrainerMineArgs
from utils import dist_sync_objects, dist_broadcast_objects
from data import DPORefModelOuts


@torch.no_grad()
def dpo_forward_ref_logp(model, encodings: dict, pad_token_id: int) -> torch.Tensor:
    model_forward_inputs = {k: encodings[k] for k in ['input_ids', 'attention_mask']}
    logits = model(**model_forward_inputs, return_dict=True)['logits'][:, :-1, :]

    # [bsz, seqlen, vocab]
    labels = encodings['labels'][:, 1:].clone()
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    loss_mask = (labels != pad_token_id).float()
    return (per_token_logps * loss_mask).sum(-1).detach()


def dpo_forward_policy_logp(model, encodings: dict, pad_token_id: int, return_mean: bool = False) -> torch.Tensor:

    model_forward_inputs = {k: encodings[k] for k in ['input_ids', 'attention_mask']}
    logits = model(**model_forward_inputs, return_dict=True)['logits'][:, :-1, :]

    # [bsz, seqlen, vocab]
    labels = encodings['labels'][:, 1:].clone()
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    loss_mask = (labels != pad_token_id).float()
    log_probs = (per_token_logps * loss_mask).sum(-1)
    if return_mean: log_probs /= loss_mask.sum(-1)

    return log_probs


class CustomTrainer(Trainer):
    def __init__(self, group_gloo, myargs: TrainConfig, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # mine object
        self.mine = TrainerMineArgs(group_gloo=group_gloo, args=myargs, dpo_ref_results=None)

        # pre compute the dpo ref logprobs
        if myargs.train_stage == TrainStageEnum.dpo:
            self.mine.dpo_ref_results = self.pre_compute_dpo_ref()
        
        # remove original on_log func
        def trivial_log_pass(*args, **kwargs): pass
        for cb in self.callback_handler.callbacks:
            if isinstance(cb, ProgressCallback):
                cb.on_log = trivial_log_pass


    def training_step(self, model: PreTrainedModel, batch: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:

        model.train()
        inputs = self._prepare_inputs(batch['tensor_inputs'])
        step = self.state.global_step + 1
        bsz = len(batch['uuid'])

        metrics = {}

        # compute loss
        with self.compute_loss_context_manager():
            if self.mine.args.train_stage == TrainStageEnum.sft:
                loss = model(**inputs.to(model.device), return_dict=True)['loss']
            
            elif self.mine.args.train_stage == TrainStageEnum.dpo:
                # policy
                selected, rejected = dpo_forward_policy_logp(
                    self.model, inputs, self.tokenizer.pad_token_id, return_mean=False
                ).split(bsz)

                # ref
                ref_results = [self.mine.dpo_ref_results[u].selected for u in batch['uuid']] + \
                      [self.mine.dpo_ref_results[u].rejected for u in batch['uuid']]
                ref_selected, ref_rejected = torch.tensor(ref_results, device=model.device).split(bsz)

                # dpo rewards
                selected_rewards = self.mine.args.dpo.beta * (selected - ref_selected)
                rejected_rewards = self.mine.args.dpo.beta * (rejected - ref_rejected)
                dpo_logits = selected_rewards - rejected_rewards
                losses = - nn.functional.logsigmoid(dpo_logits)

                if self.mine.args.dpo.method == DPOMethodEnum.original:
                    pass
                
                elif self.mine.args.dpo.method in [DPOMethodEnum.consistency_dyn, DPOMethodEnum.consistency_avg]:
                    # whatever the avg or dyn, the consistency rates have been already processed in the data loading stage
                    consistency = torch.tensor(batch['consistency'], device=model.device)
                    reverse_dpo_logits = rejected_rewards - selected_rewards
                    losses = consistency * losses + (1 - consistency) * (-nn.functional.logsigmoid(reverse_dpo_logits))

                    metrics['dpo_consistency'] = consistency
                    metrics['dpo_reverse_logits'] = reverse_dpo_logits

                
                else:
                    raise NotImplementedError(f"unsupported dpo method: {self.mine.args.dpo.method}")

                
                metrics["dpo_rewards_selected"] = selected_rewards
                metrics["dpo_rewards_rejected"] = rejected_rewards
                metrics["dpo_logits"] = dpo_logits
                metrics["dpo_rewards_accuracy"] = (dpo_logits > 0).float()
                metrics["loss"] = losses

                loss = losses.mean()

            else:
                raise NotImplementedError()

        total_loss = self.process_loss(loss)
        metrics = {k: sum(v.tolist()) / len(v) for k, v in metrics.items()}
        self.log(metrics, is_print=False)

        return total_loss.detach()

    @torch.no_grad()
    def evaluate(
        self,
        *args,
        **kwargs,
    ) -> Dict[str, float]:
        return {}
    

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        from transformers.trainer import TRAINING_ARGS_NAME, PREFIX_CHECKPOINT_DIR

        output_dir = output_dir if output_dir is not None else self.args.output_dir

        dst_dir = output_dir
        if output_dir.endswith('checkpoint-last'):
            if self.state.global_step != self.state.max_steps:
                tools_log_on_rank(f"saving checkpoint-last, but current step={self.state.global_step} != max_step {self.state.max_steps}", level='error')
            output_dir = f"{self.args.output_dir}/{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        os.makedirs(output_dir, exist_ok=True)
        if dst_dir != output_dir:
            os.symlink(os.path.basename(output_dir), dst_dir, target_is_directory=True)

        tools_log_on_rank(f"Saving model checkpoint to {output_dir}")

        self.model.save_pretrained(
            is_main_process=self.state.is_world_process_zero, max_shard_size="10GB", save_peft_format=self.mine.args.lora.enable,
            save_directory=output_dir, state_dict=state_dict,
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
    
    def process_loss(self, loss: torch.Tensor) -> torch.Tensor:
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # loss.backward()
        self.accelerator.backward(loss)
        
        return loss / self.args.gradient_accumulation_steps

    def log(self, logs: Dict[str, float], is_print=True) -> None:
        """
        Log `logs` on the various objects watching training.

        Subclass and override this method to inject custom behavior.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
        """
        if len(logs) == 0: return self.control

        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        output = {**logs, **{"step": self.state.global_step}}

        self.state.log_history.append(output)

        if is_print:
            tools_log_on_rank(output)
        
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)

    @torch.no_grad()
    def pre_compute_dpo_ref(self):
        """
        pre compute the dpo ref model log_probs
        """
        # lora enabled
        if self.model._hf_peft_config_loaded:
            self.model.disable_adapters()

        self.model.eval()
        ref_results = {}

        def cal_ref(batch: dict):
            uuids = batch['uuid']
            if all(u in ref_results for u in uuids): return
            bsz = len(uuids)

            selected, rejected = dpo_forward_ref_logp(
                self.model, batch['tensor_inputs'].to(self.model.device), self.tokenizer.pad_token_id
            ).split(bsz)
            
            for uuid, s, r in zip(uuids, selected, rejected):
                ref_results[temp.uuid] = (temp := DPORefModelOuts(
                    uuid, s, r
                ))

        self._train_batch_size *= 1
        for batch in tqdm(self.get_train_dataloader(), disable=self.mine.args.common.rank != 0, desc="dpo ref pre-computing-train-set"):
            cal_ref(batch)
        self._train_batch_size //= 1
            
        # sync results
        ref_results = dist_sync_objects(ref_results, self.mine.group_gloo, self.mine.args.common.rank, self.mine.args.common.world_size, dedup_key='uuid')
        ref_results = dist_broadcast_objects(ref_results, self.mine.args.common.rank, self.mine.group_gloo)

        # re-enable the adapter
        if self.model._hf_peft_config_loaded:
            self.model.set_adapter("default")

        self.accelerator.free_memory()
        self.model.train()
        return ref_results