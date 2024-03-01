import functools
import json
import logging
import operator
import os
from typing import Tuple

import colossalai
import torch
import torch.distributed as dist
import torch.nn as nn
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from opendit.dit.dit import DiT
from opendit.utils.operation import model_sharding


def load_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, file_path: str):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def remove_padding(tensor: torch.Tensor, original_shape: Tuple) -> torch.Tensor:
    return tensor[: functools.reduce(operator.mul, original_shape)]


def model_gathering(model: torch.nn.Module, model_shape_dict: dict):
    global_rank = dist.get_rank()
    global_size = dist.get_world_size()
    for name, param in model.named_parameters():
        all_params = [torch.empty_like(param.data) for _ in range(global_size)]
        dist.all_gather(all_params, param.data, group=dist.group.WORLD)
        if global_rank == 0:
            all_params = torch.cat(all_params)
            param.data = remove_padding(all_params, model_shape_dict[name]).view(model_shape_dict[name])
    dist.barrier()


def record_model_param_shape(model: torch.nn.Module) -> dict:
    param_shape = {}
    for name, param in model.named_parameters():
        param_shape[name] = param.shape
    return param_shape


def save(
    booster: Booster,
    model: nn.Module,
    ema: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    epoch: int,
    step: int,
    global_step: int,
    batch_size: int,
    coordinator: DistCoordinator,
    save_dir: str,
    shape_dict: dict,
    sequence_parallel_type: str,
    shard_ema: bool = False,
):
    torch.cuda.empty_cache()
    global_rank = dist.get_rank()
    save_dir = os.path.join(save_dir, f"epoch{epoch}-global_step{global_step}")
    os.makedirs(os.path.join(save_dir, "model"), exist_ok=True)
    # Rearrange the qkv projection (qkv ... qkv -> q ... q k ... k v ... v) in DiT model when using longseq sequence parallelism
    if global_rank == 0 and sequence_parallel_type == "longseq":
        if isinstance(model.module, DiT):
            model.module.rearrange_attention_weights(flag="save")
        else:
            model.module.module.rearrange_attention_weights(flag="save")
    booster.save_model(model, os.path.join(save_dir, "model"), shard=True)

    # Gather the sharded ema model before saving
    if shard_ema:
        model_gathering(ema, shape_dict)
    # Rearrange the qkv projection (qkv ... qkv -> q ... q k ... k v ... v) in ema model when using longseq sequence parallelism
    if global_rank == 0 and sequence_parallel_type == "longseq":
        ema.rearrange_attention_weights(flag="save")
    # ema is not boosted, so we don't need to use booster.save_model
    if global_rank == 0:
        torch.save(ema.state_dict(), os.path.join(save_dir, "ema.pt"))
        # Shard ema model when using zero2 plugin
        if shard_ema:
            model_sharding(ema)
    if optimizer is not None:
        booster.save_optimizer(optimizer, os.path.join(save_dir, "optimizer"), shard=True, size_per_shard=4096)
    if lr_scheduler is not None:
        booster.save_lr_scheduler(lr_scheduler, os.path.join(save_dir, "lr_scheduler"))
    running_states = {
        "epoch": epoch,
        "step": step,
        "global_step": global_step,
        "sample_start_index": step * batch_size,
    }
    if coordinator.is_master():
        save_json(running_states, os.path.join(save_dir, "running_states.json"))
    dist.barrier()


def load(
    booster: Booster,
    model: nn.Module,
    ema: nn.Module,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    load_dir: str,
    sequence_parallel_type: str,
) -> Tuple[int, int, int]:
    booster.load_model(model, os.path.join(load_dir, "model"))
    # ema is not boosted, so we don't use booster.load_model
    ema.load_state_dict(torch.load(os.path.join(load_dir, "ema.pt"), map_location=torch.device("cpu")))
    if optimizer is not None:
        booster.load_optimizer(optimizer, os.path.join(load_dir, "optimizer"))
    if lr_scheduler is not None:
        booster.load_lr_scheduler(lr_scheduler, os.path.join(load_dir, "lr_scheduler"))
    running_states = load_json(os.path.join(load_dir, "running_states.json"))
    # Rearrange the qkv projection (q ... q k ... k v ... v -> qkv ... qkv ) in when using longseq sequence parallelism
    if sequence_parallel_type == "longseq":
        if isinstance(model.module, DiT):
            model.module.rearrange_attention_weights()
        else:
            model.module.module.rearrange_attention_weights()
        ema.rearrange_attention_weights()
    dist.barrier()
    torch.cuda.empty_cache()
    return running_states["epoch"], running_states["step"], running_states["sample_start_index"]


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def load_from_sharded_state_dict(model, ckpt_path):
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    colossalai.launch_from_torch({})
    plugin = LowLevelZeroPlugin(
        stage=2,
        precision="fp32",
        initial_scale=2**16,
    )
    booster = Booster(plugin=plugin)
    model, _, _, _, _ = booster.boost(model=model)
    booster.load_model(model, os.path.join(ckpt_path, "model"))
