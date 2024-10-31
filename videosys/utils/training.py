import math
import random
import time
from collections import OrderedDict
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.cluster.dist_coordinator import DistCoordinator
from colossalai.zero.low_level.low_level_optim import LowLevelZeroOptimizer
from opendit.core.parallel_mgr import set_parallel_manager

from videosys.utils.logging import logger


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


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, optimizer=None, decay: float = 0.9999, sharded: bool = True
) -> None:
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name == "pos_embed":
            continue
        if not param.requires_grad:
            continue
        if not sharded:
            param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)
        else:
            if param.data.dtype != torch.float32:
                param_id = id(param)
                master_param = optimizer._param_store.working_to_master_param[param_id]
                param_data = master_param.data
            else:
                param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)


class MaskGenerator:
    def __init__(self, mask_ratios):
        valid_mask_names = [
            "identity",
            "quarter_random",
            "quarter_head",
            "quarter_tail",
            "quarter_head_tail",
            "image_random",
            "image_head",
            "image_tail",
            "image_head_tail",
            "random",
            "intepolate",
        ]
        assert all(
            mask_name in valid_mask_names for mask_name in mask_ratios.keys()
        ), f"mask_name should be one of {valid_mask_names}, got {mask_ratios.keys()}"
        assert all(
            mask_ratio >= 0 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be greater than or equal to 0, got {mask_ratios.values()}"
        assert all(
            mask_ratio <= 1 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be less than or equal to 1, got {mask_ratios.values()}"
        # sum of mask_ratios should be 1
        if "identity" not in mask_ratios:
            mask_ratios["identity"] = 1.0 - sum(mask_ratios.values())
        assert math.isclose(
            sum(mask_ratios.values()), 1.0, abs_tol=1e-6
        ), f"sum of mask_ratios should be 1, got {sum(mask_ratios.values())}"
        logger.info("mask ratios: %s", mask_ratios)
        self.mask_ratios = mask_ratios

    def get_mask(self, x):
        mask_type = random.random()
        mask_name = None
        prob_acc = 0.0
        for mask, mask_ratio in self.mask_ratios.items():
            prob_acc += mask_ratio
            if mask_type < prob_acc:
                mask_name = mask
                break

        num_frames = x.shape[2]
        # Hardcoded condition_frames
        condition_frames_max = num_frames // 4

        mask = torch.ones(num_frames, dtype=torch.bool, device=x.device)
        if num_frames <= 1:
            return mask

        if mask_name == "quarter_random":
            random_size = random.randint(1, condition_frames_max)
            random_pos = random.randint(0, x.shape[2] - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "image_random":
            random_size = 1
            random_pos = random.randint(0, x.shape[2] - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "quarter_head":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
        elif mask_name == "image_head":
            random_size = 1
            mask[:random_size] = 0
        elif mask_name == "quarter_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[-random_size:] = 0
        elif mask_name == "image_tail":
            random_size = 1
            mask[-random_size:] = 0
        elif mask_name == "quarter_head_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "image_head_tail":
            random_size = 1
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "intepolate":
            random_start = random.randint(0, 1)
            mask[random_start::2] = 0
        elif mask_name == "random":
            mask_ratio = random.uniform(0.1, 0.9)
            mask = torch.rand(num_frames, device=x.device) > mask_ratio
            # if mask is all False, set the last frame to True
            if not mask.any():
                mask[-1] = 1

        return mask

    def get_masks(self, x):
        masks = []
        for _ in range(len(x)):
            mask = self.get_mask(x)
            masks.append(mask)
        masks = torch.stack(masks, dim=0)
        return masks


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, optimizer=None, decay: float = 0.9999, sharded: bool = True
) -> None:
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name == "pos_embed":
            continue
        if param.requires_grad == False:
            continue
        if not sharded:
            param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)
        else:
            if param.data.dtype != torch.float32 and isinstance(optimizer, LowLevelZeroOptimizer):
                param_id = id(param)
                master_param = optimizer._param_store.working_to_master_param[param_id]
                param_data = master_param.data
            else:
                param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)


def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


class Timer:
    def __init__(self, name, log=False, coordinator: Optional[DistCoordinator] = None):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.log = log
        self.coordinator = coordinator

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
        if self.coordinator is not None:
            self.coordinator.block_all()
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


def create_colossalai_plugin(plugin, dtype, grad_clip, sp_size, reduce_bucket_size_in_m: int = 20):
    if plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=dtype,
            initial_scale=2**16,
            max_norm=grad_clip,
            reduce_bucket_size_in_m=reduce_bucket_size_in_m,
        )
        set_parallel_manager(dist.get_world_size() // sp_size, sp_size)
    else:
        raise ValueError(f"Unknown plugin {plugin}")
    return plugin


def set_grad_accumulation_steps(engine, total_gas: int):
    """
    A special hack to enable automatic gradient accumulation with deepspeed engine.
    """
    engine.set_train_batch_size(engine.train_micro_batch_size_per_gpu() * engine.dp_world_size * total_gas)
    engine.optimizer.gradient_accumulation_steps = total_gas
