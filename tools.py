import time
import json
import os
import sys
import inspect
import warnings

from loguru import logger
from datetime import datetime


ABSOLUTE_WRONG_FINAL_ANS = "will_never_be_the_right_answer"

logger.remove()
loguru_format = "<green>{extra[caller_time]} (UTC+8)</green> | <level>{level:<8}</level> | " + \
"<cyan>{extra[caller_file]}:{extra[caller_module]}:{extra[caller_function]}:{extra[caller_line]}</cyan> -\n" + \
"<level>{message}</level>" + f"\n<red>{'^'*30}</red>"

logger.add(sys.stdout, colorize=True, format=loguru_format, level="INFO")

def warning_to_loguru(message, category, filename, lineno, file=None, line=None):
    """Redirect warning to Loguru logger."""
    tools_log_on_rank(message, level='warning', filename=filename, lineno=lineno)

# Redirect all warnings to the custom handler
warnings.simplefilter("ignore", FutureWarning)
warnings.showwarning = warning_to_loguru


def tools_is_device_cpu(device: str) -> bool:
    """check whether the device is in cpu mode; you can pass the device like 0,1,2,3 or cpu or cpu,cpu"""
    if device == 'cpu': return True
    temp = [x.isdigit() for x in device.split(',')]
    assert all(temp) is True or any(temp) is False, f"invalid devices = {device} = {temp}"
    return not temp[0]

def tools_get_random_available_port():
    import socket
    from contextlib import closing
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('localhost', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        port = s.getsockname()[1]
    time.sleep(3)
    return port

def tools_json_load(path) -> dict | list:
    with open(path, 'r') as file:
        return json.load(file)

def tools_json_dump(obj, path):
    with open(path, 'w') as file:
        json.dump(obj, file, indent=4)

def tools_get_model_name(load: str):
    if '/' in load:
        return load.split('/')[-1]
    else:
        return load

def tools_set_device_env(device: str):
    if tools_is_device_cpu(device):
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ['ROCR_VISIBLE_DEVICES'] = ""
    else:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = device
        import torch
        assert torch.cuda.device_count() > 0, "No GPU available!"
            

def tools_print_trainable_params(model):
    """
    Prints the number of trainable parameters in the model.
    """

    def get_nb_trainable_parameters(self):
        r"""
        Returns the number of trainable parameters and number of all parameters in the model.
        """
        # note: same as PeftModel.get_nb_trainable_parameters
        trainable_params = 0
        all_param = 0
        for _, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            # Due to the design of 4bit linear layers from bitsandbytes
            # one needs to multiply the number of parameters by 2 to get
            # the correct number of parameters
            if param.__class__.__name__ == "Params4bit":
                num_params = num_params * 2

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param

    trainable_params, all_param = get_nb_trainable_parameters(model)

    tools_log_on_rank(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_param:,d} || "
        f"trainable%: {100 * trainable_params / all_param:.4f}"
    )

def tools_get_simple_dataset_name(name: str) -> str:
    if name.endswith(".json") or name.endswith(".jsonl"):
        name = os.path.dirname(name)
    return os.path.basename(name)

def tools_get_time() -> str:
    import pytz
    ZONE = pytz.timezone("Asia/Chongqing")
    return datetime.now(ZONE).strftime("%y-%m-%d-%H_%M_%S")

def tools_elapsed_time(previous_time_str: str) -> str:
    previous_dt = datetime.strptime(previous_time_str, "%y-%m-%d-%H_%M_%S")
    current_dt = datetime.strptime(tools_get_time(), "%y-%m-%d-%H_%M_%S")
    delta = current_dt - previous_dt
    days = delta.days
    seconds = delta.seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 3600) % 60
    
    return f"{days} days, {hours} hours, {minutes} minutes, {seconds} seconds"

def tools_log_on_rank(*msgs, rank: int | None = 0, level='INFO', filename=None, lineno=None):
    """rank=None: all ranks will log this message"""
    if os.environ.get('DISABLE_LOG') == '1': return
    level = level.upper()
    cur_rank = os.environ.get('RANK')
    msg = ' '.join([str(msg) for msg in msgs])

    
    if filename and lineno:
        caller_info =  {
            "caller_file": filename,
            "caller_module": '<unknown>',
            "caller_function": '<unknown>',
            "caller_line": lineno,
        }
    else:
        caller_frame = inspect.stack()[1]
        caller_info = {
            "caller_file": os.path.basename(caller_frame.filename),
            "caller_module": caller_frame.frame.f_globals.get('__name__', '<unknown>'),
            "caller_function": caller_frame.function,
            "caller_line": caller_frame.lineno,
        }

    caller_info['caller_time'] = tools_get_time()

    if cur_rank is not None: cur_rank = int(cur_rank)
    if rank is None or cur_rank is None or cur_rank == rank:
        logger.bind(**caller_info).log(level, msg)
    else:
        return
    

def tools_is_lora_ckpt(checkpoint: str | None) -> bool:
    if checkpoint is None: return False
    assert os.path.isdir(checkpoint) or os.path.islink(checkpoint), f"Invalid checkpoint: {checkpoint}"
    if os.path.exists(f"{checkpoint}/adapter_model.safetensors"):
        is_lora = True
    elif os.path.exists(f"{checkpoint}/model.safetensors"):
        # fix patch
        if os.path.exists(index_file := f"{checkpoint}/model.safetensors.index.json"):
            os.rename(index_file, f"{checkpoint}/remove_me.model.safetensors.index.json")
        is_lora = False
    elif os.path.exists(index_file := f"{checkpoint}/model.safetensors.index.json"):
        is_lora = False
    else:
        raise ValueError(f"Invalid checkpoint: {checkpoint}")

    return is_lora

def tools_assert_ckpt_name_valid(checkpoint: str | None) -> bool:
    """check checkpoint name is valid in configs/checkpoint.json"""
    if checkpoint is None:
        pass
    else:
        ckpt_map_path = 'configs/checkpoint.json'
        ckpt_map = tools_json_load(ckpt_map_path)
        assert checkpoint in ckpt_map, f"checkpoint={checkpoint} is not in {ckpt_map.keys()}, please check {ckpt_map_path}"
        assert os.path.exists(ckpt_map[checkpoint]), f"checkpoint={ckpt_map[checkpoint]} does not exist"
        assert len(set(ckpt_map.values())) == len(ckpt_map.values()), "checkpoint path should be unique"

def tools_get_checkpoint_load_path(checkpoint: str | None) -> str | None:
    if checkpoint is None: return None
    from tools import tools_json_load
    ckpt_map = tools_json_load('configs/checkpoint.json')
    return ckpt_map[checkpoint]