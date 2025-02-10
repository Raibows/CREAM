from config import InferVllmConfig, parse_args, InferModeEnum, DatasetEnum
from tools import tools_set_device_env, tools_json_load, tools_json_dump, tools_get_time, tools_log_on_rank, tools_get_checkpoint_load_path, tools_is_lora_ckpt, tools_elapsed_time
from evaluate import extract_final_answer
from prompts import get_sys_prompt

import os
import copy




class LLMPredictor:

    def __init__(self, is_lora: bool, args: InferVllmConfig, sampling_args, lora_ckpt_path: str | None):
        # Create an LLM.
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        self.llm = LLM(
            model=args.model.value if (is_lora or args.checkpoint is None) else args.checkpoint,
            tokenizer=args.model.value,
            trust_remote_code=True,
            tensor_parallel_size=args.tensor_parallel_size,
            dtype='auto',
            enable_lora=is_lora,
            max_lora_rank=64,
            # input len + gen max len
            max_model_len=args.dataset.max_length + args.generation.gen_max_tokens,
            load_format='dummy' if args.common.debug else 'auto',
        )
        self.sampling_args = sampling_args

        if lora_ckpt_path:
            self.lora_request = LoRARequest(lora_ckpt_path, 1, lora_ckpt_path)
        else:
            self.lora_request = None

    def __call__(self, batch: list[str]) -> dict[str, list]:
        outputs = self.llm.generate(batch['prompt'], self.sampling_args, lora_request=self.lora_request)
        prompt: list[str] = []
        generated_text: list[list[str]] = []

        for output in outputs:
            prompt.append(output.prompt)
            generated_text.append([o.text for o in output.outputs])

        return {
            "uuid": batch['uuid'],
            "prompt": prompt,
            "responses": generated_text,
        }

class Scheduler:
    def __init__(self, tp_size: int):
        self.tp_size = tp_size
    
    def __call__(self,):
        from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
        import ray
        pg = ray.util.placement_group(
            [{
                "GPU": 1,
                "CPU": 2
            }] * self.tp_size,
            strategy="STRICT_PACK",
        )
        pg = PlacementGroupSchedulingStrategy(pg, placement_group_capture_child_tasks=True)
        return dict(scheduling_strategy=pg)
        
        

def main(time_based: str, args: InferVllmConfig):
    from typing import Any, Dict, List
    import numpy as np
    import ray
    from packaging.version import Version
    import ray.data
    
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    assert Version(ray.__version__) >= Version("2.22.0"), "Ray version must be at least 2.22.0"

    # read data
    data = tools_json_load(args.dataset.name.value)
    sys_prompt = get_sys_prompt(args.sysprompt)
    tools_log_on_rank(f"sys_prompt={sys_prompt}")

    if args.mode == InferModeEnum.sampling:
        # sysprompt
        data: dict
        for k, v in data.items():
            data[k][0]['content'] = data[k][0]['content'] + sys_prompt

        ori_data = copy.deepcopy(data)
        keys = list(data.keys())
        data = list(data.values())

    elif args.mode == InferModeEnum.evaluation:
        data: list[dict]
        # sysprompt
        for i in range(len(data)):
            data[i]['question'] = data[i]['question'] + sys_prompt

        ori_data = {item['uuid']: item for item in data}
        keys = [item['uuid'] for item in data]
        data = [
            [{"role": "user", "content": item['question']}]
            for item in data
        ]

    else:
        raise NotImplementedError(f"mode={args.mode}")


    # debug
    tools_log_on_rank(f"Data length={len(data)}, check the examples\n{data[0]}")
    if args.common.debug:
        data = data[:97]

    tokenizer = AutoTokenizer.from_pretrained(args.model.value)
    data = tokenizer.apply_chat_template(data, padding=False, truncation=False, tokenize=False, add_generation_prompt=True)
    data = [{'uuid': k, 'prompt': v} for k, v in zip(keys, data)]
    
    print(data[0])

    sampling_params = SamplingParams(
        n=args.generation.gen_n, 
        top_p=args.generation.top_p,
        temperature=args.generation.gen_temperature,
        max_tokens=args.generation.gen_max_tokens,
        truncate_prompt_tokens=args.dataset.max_length,
    )

    
    # test is lora or full ckpt
    args.checkpoint = tools_get_checkpoint_load_path(args.checkpoint)
    is_lora = tools_is_lora_ckpt(args.checkpoint)

    # configure ray gpus
    resources_kwarg: Dict[str, Any] = {}
    if args.tensor_parallel_size == 1:
        resources_kwarg["num_gpus"] = 1
    else:
        resources_kwarg["num_gpus"] = 0
        resources_kwarg["ray_remote_args_fn"] = Scheduler(args.tensor_parallel_size)


    # run
    ds = ray.data.from_items(data)

    dp_size = args.common.world_size // args.tensor_parallel_size
    # Apply batch inference for all input data.
    ds = ds.map_batches(
        LLMPredictor,
        fn_constructor_args=[is_lora, args, sampling_params, args.checkpoint if is_lora else None],
        concurrency=dp_size,
        # Specify the batch size for inference.
        batch_size=len(data) // dp_size,
        **resources_kwarg,
    )

    # write results
    results = {}
    outputs = ds.take_all()

    if args.mode == InferModeEnum.sampling:
        for output in outputs:
            uuid = output["uuid"]
            results[uuid] = {
                'prompt': ori_data[uuid],
                'responses': output['responses']
            }
    
    elif args.mode == InferModeEnum.evaluation:
        uuid2task = {}
        for task in args.dataset.get_downstream_tasks():
            for item in tools_json_load(task.value):
                uuid2task[item['uuid']] = task

        for output in outputs:
            uuid = output["uuid"]
            task = uuid2task[uuid]
            assert task == args.dataset.name or args.dataset.name == DatasetEnum.all_test

            results[uuid] = {
                **ori_data[uuid],
                'responses': output['responses'][0],
                'final_prediction': (pred := extract_final_answer(output['responses'][0], task)),
                'correct': pred.lower() == str(ori_data[uuid]['label']).lower()
            }


    if args.mode == InferModeEnum.sampling:
        output_path = f"{args.common.output_dir}/results.json"

    elif args.mode == InferModeEnum.evaluation:
        all_acc = [
            [item['correct'] for uuid, item in results.items() if uuid2task[uuid] == k]
            for k in args.dataset.get_downstream_tasks()
        ]
        all_acc = [f"{sum(acc) / len(acc) * 100 :.3f}" if len(acc) > 0 else None for acc in all_acc]

        acc_str = "resultsAcc"
        for task, acc in zip(args.dataset.get_downstream_tasks(), all_acc):
            if acc is not None:
                tools_log_on_rank(f"dataset={task.name}, model = {args.model.name}, ckpt = {args.checkpoint}, acc = {acc}")
                acc_str = f"{acc_str}-{task.name}={acc}"

        output_path = f"{args.common.output_dir}/{acc_str}.json"

    if os.path.exists(output_path):
        output_path = output_path[:-5] + f"_{time_based}.json"
    
    tools_json_dump(results, output_path)
    tools_log_on_rank(f"Results are saved to={output_path}, costs {tools_elapsed_time(time_based)}")




if __name__ == "__main__":
    args: InferVllmConfig = parse_args(InferVllmConfig, pass_in=[])
    tools_set_device_env(args.common.device)
    time_based = tools_get_time()
    suffix = f"{args.dataset.name.name}-{args.model.name}-ckpt_{args.checkpoint}-{args.generation}-sys_{args.sysprompt.name}"
    if args.common.debug:
        args.common.output_dir = f"outputs/debug/{args.mode.value}"
    else:
        args.common.output_dir = f"outputs/{args.mode.value}"
    
    args.common.output_dir = f"{args.common.output_dir}/{suffix}"
    os.makedirs(args.common.output_dir, exist_ok=True)
    
    args_path = f"{args.common.output_dir}/args.json"
    args.save_json(args_path, indent=4)
    
    print(args)
    tools_log_on_rank(f"the output dir={args.common.output_dir}")
    main(time_based, args)
