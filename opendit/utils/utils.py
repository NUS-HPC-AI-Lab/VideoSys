import logging
import os
import random

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


def get_logger():
    return logging.getLogger(__name__)


def create_logger(logging_dir=None):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger # BUG 
    # if True:  # real logger
        additional_args = dict()
        if logging_dir is not None:
            additional_args["handlers"] = [
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ]
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            **additional_args,
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger



def space_timesteps(time_steps, time_bins):
    num_bins = len(time_bins)
    bin_size = time_steps // num_bins
    
    result = []
    
    for i, bin_count in enumerate(time_bins):
        start = i * bin_size
        end = start + bin_size
        
        bin_steps = np.linspace(start, end, bin_count, endpoint=False, dtype=int).tolist()
        result.extend(bin_steps)
    
    result_tensor = torch.tensor(result, dtype=torch.int32)
    sorted_tensor = torch.sort(result_tensor, descending=True).values
    
    return sorted_tensor



def skip_diffusion_timestep(timesteps, diffusion_skip_timestep):
    if isinstance(timesteps, list):
        # If timesteps is a list, we assume each element is a tensor
        timesteps_np = [t.cpu().numpy() for t in timesteps]
        device = timesteps[0].device
    else:
        # If timesteps is a tensor
        timesteps_np = timesteps.cpu().numpy()
        device = timesteps.device

    num_bins = len(diffusion_skip_timestep)
    
    if isinstance(timesteps_np, list):
        bin_size = len(timesteps_np) // num_bins
        new_timesteps = []
        
        for i in range(num_bins):
            bin_start = i * bin_size
            bin_end = (i + 1) * bin_size if i != num_bins - 1 else len(timesteps_np)
            bin_timesteps = timesteps_np[bin_start:bin_end]
            
            if diffusion_skip_timestep[i] == 0:
                # If the bin is marked with 0, keep all timesteps
                new_timesteps.extend(bin_timesteps)
            elif diffusion_skip_timestep[i] == 1:
                # If the bin is marked with 1, omit the last timestep in the bin
                new_timesteps.extend(bin_timesteps[1:])
        
        new_timesteps_tensor = [torch.tensor(t, device=device) for t in new_timesteps]
    else:
        bin_size = len(timesteps_np) // num_bins
        new_timesteps = []
        
        for i in range(num_bins):
            bin_start = i * bin_size
            bin_end = (i + 1) * bin_size if i != num_bins - 1 else len(timesteps_np)
            bin_timesteps = timesteps_np[bin_start:bin_end]
            
            if diffusion_skip_timestep[i] == 0:
                # If the bin is marked with 0, keep all timesteps
                new_timesteps.extend(bin_timesteps)
            elif diffusion_skip_timestep[i] == 1:
                # If the bin is marked with 1, omit the last timestep in the bin
                new_timesteps.extend(bin_timesteps[1:])
        
        new_timesteps_tensor = torch.tensor(new_timesteps, device=device)
    
    if isinstance(timesteps, list):
        return new_timesteps_tensor
    else:
        return new_timesteps_tensor