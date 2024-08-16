import os
import random
import socket

import imageio
import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig, OmegaConf


def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def str_to_dtype(x: str):
    if x == "fp32":
        return torch.float32
    elif x == "fp16":
        return torch.float16
    elif x == "bf16":
        return torch.bfloat16
    else:
        raise RuntimeError(f"Only fp32, fp16 and bf16 are supported, but got {x}")


def merge_args(args1, args2):
    """
    Merge two argparse Namespace objects.
    """
    if args2 is None:
        return args1

    for k in args2._content.keys():
        if k in args1.__dict__:
            v = getattr(args2, k)
            if isinstance(v, ListConfig) or isinstance(v, DictConfig):
                v = OmegaConf.to_object(v)
            setattr(args1, k, v)
        else:
            raise RuntimeError(f"Unknown argument {k}")

    return args1


def all_exists(paths):
    return all(os.path.exists(path) for path in paths)


def save_video(video, output_path, fps):
    """
    Save a video to disk.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if dist.get_rank() == 0:
        imageio.mimwrite(output_path, video, fps=fps)
    dist.barrier()


def get_distributed_init_method(ip: str, port: int) -> str:
    # Brackets are not permitted in ipv4 addresses,
    # see https://github.com/python/cpython/issues/103848
    return f"tcp://[{ip}]:{port}" if ":" in ip else f"tcp://{ip}:{port}"


def get_open_port() -> int:
    # try ipv4
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
    except OSError:
        # try ipv6
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]
