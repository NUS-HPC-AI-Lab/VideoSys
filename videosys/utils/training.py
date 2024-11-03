import glob
import os
import time
from typing import Tuple

import torch
import torch.distributed as dist


def get_model_numel(model: torch.nn.Module) -> Tuple[int, int]:
    num_params = 0
    num_params_trainable = 0
    for p in model.parameters():
        num_params += p.numel()
        if p.requires_grad:
            num_params_trainable += p.numel()
    return num_params, num_params_trainable


def format_numel_str(numel: int) -> str:
    B = 1024**3
    M = 1024**2
    K = 1024
    if numel >= B:
        return f"{numel / B:.2f} B"
    elif numel >= M:
        return f"{numel / M:.2f} M"
    elif numel >= K:
        return f"{numel / K:.2f} K"
    else:
        return f"{numel}"


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    dist.all_reduce(tensor=tensor, op=dist.ReduceOp.SUM)
    tensor.div_(dist.get_world_size())
    return tensor


def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def define_experiment_workspace(outputs, get_last_workspace=False):
    """
    This function creates a folder for experiment tracking.

    Args:
        args: The parsed arguments.

    Returns:
        exp_dir: The path to the experiment folder.
    """
    # Make outputs folder (holds all experiment subfolders)
    os.makedirs(outputs, exist_ok=True)
    experiment_index = len(glob.glob(f"{outputs}/*"))
    if get_last_workspace:
        experiment_index -= 1

    # Create an experiment folder
    model_name = "OpenSora"
    exp_name = f"{experiment_index:03d}-{model_name}"
    exp_dir = f"{outputs}/{exp_name}"
    return exp_name, exp_dir


class Timer:
    def __init__(self, name, log=False):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.log = log

        self.start_mem = None
        self.end_mem = None

    @property
    def elapsed_time(self):
        return self.end_time - self.start_time

    def __enter__(self):
        torch.cuda.synchronize()
        if self.log:
            self.start_mem = (
                torch.cuda.memory_allocated() / 1024**3,
                torch.cuda.memory_reserved() / 1024**3,
                torch.cuda.max_memory_allocated() / 1024**3,
                torch.cuda.max_memory_reserved() / 1024**3,
            )
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        self.end_time = time.time()
        if self.log:
            self.end_mem = (
                torch.cuda.memory_allocated() / 1024**3,
                torch.cuda.memory_reserved() / 1024**3,
                torch.cuda.max_memory_allocated() / 1024**3,
                torch.cuda.max_memory_reserved() / 1024**3,
            )
            print(
                f"Elapsed time for {self.name}: {self.elapsed_time:.2f} s. "
                f" >>> [{self.name}]: allocated memory: {self.start_mem[0]:.2f} -> {self.end_mem[0]:.2f}, "
                f"reserved memory: {self.start_mem[1]:.2f} -> {self.end_mem[1]:.2f}, "
                f"max allocated memory: {self.start_mem[2]:.2f} -> {self.end_mem[2]:.2f}, "
                f"max reserved memory: {self.start_mem[3]:.2f} -> {self.end_mem[3]:.2f}"
            )


class GroupTimer(Timer):
    def __init__(self, name, log=False, group=None):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.log = log
        self.group = group
        self.sync_tensor = torch.tensor(
            [0, 0], dtype=torch.int, device=torch.device(f"cuda:{torch.cuda.current_device()}")
        )

        self.start_mem = None
        self.end_mem = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.group is not None:
            dist.all_reduce(self.sync_tensor, group=self.group)
        torch.cuda.synchronize()
        self.end_time = time.time()
        if self.log:
            self.end_mem = (
                torch.cuda.memory_allocated() / 1024**3,
                torch.cuda.memory_reserved() / 1024**3,
                torch.cuda.max_memory_allocated() / 1024**3,
                torch.cuda.max_memory_reserved() / 1024**3,
            )
            print(
                f">>> [{self.name} Rank {dist.get_rank()}] Elapsed time: {self.elapsed_time:.2f} s. "
                f"Alloc memory: {self.start_mem[0]:.2f} -> {self.end_mem[0]:.2f}, "
                f"Rsrvd memory: {self.start_mem[1]:.2f} -> {self.end_mem[1]:.2f}, "
                f"Max Alloc memory: {self.start_mem[2]:.2f} -> {self.end_mem[2]:.2f}, "
                f"Max Rsrvd memory: {self.start_mem[3]:.2f} -> {self.end_mem[3]:.2f}"
            )


def set_grad_accumulation_steps(engine, total_gas: int):
    """
    A special hack to enable automatic gradient accumulation with deepspeed engine.
    """
    engine.set_train_batch_size(engine.train_micro_batch_size_per_gpu() * engine.dp_world_size * total_gas)
    engine.optimizer.gradient_accumulation_steps = total_gas
