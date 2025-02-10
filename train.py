import os

from config import TrainConfig, parse_args, get_hf_training_args
from tools import tools_set_device_env, tools_get_time, tools_elapsed_time, tools_log_on_rank, tools_get_checkpoint_load_path

   
def main(localrank: int, time_based: str, args: TrainConfig):
    import torch.distributed as dist
    import os
    import datasets
    from datetime import timedelta
    from transformers import set_seed, AutoTokenizer

    from utils import setup, prepare_model, dist_broadcast_objects, dist_sync_objects, set_pad_token
    from data import PaddingCollator, preprocess_dataset
    from sft_dpo_trainer import CustomTrainer
    from trainer_utils import CustomCallback

    # init
    args.common.rank = localrank
    setup(args.common)
    set_seed(0)
    group_gloo = dist.new_group(backend="gloo")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model.value)
    set_pad_token(tokenizer)

    # dataset
    dataset = preprocess_dataset(args.dataset.name, args.train_stage, tokenizer, args.dataset.max_length, dpo_consistency_strategy=args.dpo.method)

    if args.dataset.limit_size is not None:
        if localrank == 0:
            dataset = dataset.shuffle().select(range(min(args.dataset.limit_size, len(dataset))))
        
    dataset = dist_broadcast_objects(dataset, localrank, group_gloo)

    tools_log_on_rank(f"train dataset={args.dataset.name}, size = {dataset.num_rows}")

    # load model
    tools_log_on_rank(f"loading model {args.model}")
    model = prepare_model(args.common.get_device(), args.model, args.lora, bf16=args.common.bf16, debug=args.common.debug, ckpt_path=tools_get_checkpoint_load_path(args.checkpoint), tokenizer=tokenizer)
    set_pad_token(tokenizer, model)

    # data collator
    collator = PaddingCollator(tokenizer, args.train_stage, max_length=args.dataset.max_length)

    # ---------------------------------- trainer --------------------------------- #
    hf_args = get_hf_training_args(
        args,
        args.common,
        do_eval=False
    )
    trainer = CustomTrainer(
        group_gloo,
        args,
        model=model,
        args=hf_args,
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=collator,
        callbacks=[CustomCallback],
        tokenizer=tokenizer
    )
    trainer.train()
    trainer.save_model(output_dir=f"{args.common.output_dir}/checkpoint-last")

    # overwrite args
    if args.common.rank <= 0:
        print(args)
        args.save_json(f"{args.common.output_dir}/00args.json", indent=4)

    dist.monitored_barrier(group_gloo, timeout=timedelta(minutes=10))
    dist.destroy_process_group()
    


if __name__ == '__main__':
    # -------------------------------- parse args -------------------------------- #

    pass_in = []
    # pass_in = '--device 6,7 --name oasst_dpo_1 --deepspeed configs/ds_stage2.json --enable no --debug yes --train_stage dpo --method consistency_avg --grad_checkpointing yes'.split()
    
    args: TrainConfig = parse_args(args_config=TrainConfig, pass_in=pass_in)
    tools_set_device_env(args.common.device)

    # ------------------------------ import package ------------------------------ #
    import torch.multiprocessing as mp


    # --------------------------------- post set --------------------------------- #
    time_based = tools_get_time()

    if args.common.debug and args.dataset.limit_size is None:
        args.dataset.limit_size = 19
    
    args.common.set_run_name_and_output_dir(time_based, args.dataset, args)

    os.makedirs(args.common.output_dir, exist_ok=True)
    args.save_json(f"{args.common.output_dir}/00args.json", indent=4)

    tools_log_on_rank(args)
    mp.spawn(main, args=(time_based, args), nprocs=args.common.world_size, join=True)
    tools_log_on_rank(args, f'\noutput_dir={args.common.output_dir}')
    tools_log_on_rank(f"task done starting from {time_based}, elapsed {tools_elapsed_time(time_based)}")

