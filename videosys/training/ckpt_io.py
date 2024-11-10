import json
import os
from typing import Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


def load_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)


def save_json(data, file_path: str):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)


def save(
    save_dir: str,
    save_optimizer: bool = False,
    model: nn.Module = None,
    ema: nn.Module = None,
    sampler=None,
    epoch: int = None,
    step: int = None,
    global_step: int = None,
    batch_size: int = None,
):
    save_dir = os.path.join(save_dir, f"epoch{epoch}-global_step{global_step}")
    os.makedirs(save_dir, exist_ok=True)

    if save_optimizer:
        # can only use deepspeed to load this checkpoint
        model.save_checkpoint(save_dir, tag=f"deepspeed_checkpint")

    if dist.get_rank() == 0:
        running_states = {
            "epoch": epoch,
            "step": step,
            "global_step": global_step,
            "batch_size": batch_size,
        }
        save_json(running_states, os.path.join(save_dir, "running_states.json"))

        if model is not None:
            model.module.save_pretrained(os.path.join(save_dir, "model"), safe_serialization=False)

        if ema is not None:
            ema.save_pretrained(os.path.join(save_dir, "ema"))

        if sampler is not None:
            # only for VariableVideoBatchSampler
            torch.save(sampler.state_dict(step), os.path.join(save_dir, "sampler"))

    dist.barrier()
    return save_dir


def load(
    load_dir: str,
    model: nn.Module = None,
    ema: nn.Module = None,
    sampler=None,
) -> Tuple[int, int, int]:
    assert os.path.exists(load_dir), f"Checkpoint directory {load_dir} does not exist"
    assert os.path.exists(os.path.join(load_dir, "running_states.json")), "running_states.json does not exist"
    running_states = load_json(os.path.join(load_dir, "running_states.json"))

    model.load_checkpoint(os.path.join(load_dir, "deepspeed_checkpoint"))

    if ema is not None:
        ema.load_pretrained(os.path.join(load_dir, "ema"))

    if sampler is not None:
        sampler.load_state_dict(torch.load(os.path.join(load_dir, "sampler")))

    dist.barrier()

    return (
        running_states["epoch"],
        running_states["step"],
    )


def save_training_config(cfg, experiment_dir):
    with open(f"{experiment_dir}/config.txt", "w") as f:
        json.dump(cfg, f, indent=4)
