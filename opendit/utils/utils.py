import os
import random

import numpy as np
import torch


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
