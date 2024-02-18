# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import torch

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import json
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from time import time

import numpy as np
import torch.distributed as dist
import tqdm
from diffusers.models import AutoencoderKL
from models.diffusion import create_diffusion
from models.dit import DiT_models
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

import colossalai
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device

#################################################################################
#                             Training Helper Functions                         #
#################################################################################


def get_model_numel(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


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
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


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


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # ==============================
    # Initialize Distributed Training
    # ==============================
    colossalai.launch_from_torch({})
    coordinator = DistCoordinator()
    rank = dist.get_rank()
    device = get_current_device()

    # ==============================
    # Setup an experiment folder
    # ==============================
    if coordinator.is_master():
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
        experiment_dir = f"{args.results_dir}/{experiment_index:03d}-{model_string_name}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(f"{experiment_dir}/config.txt", "w") as f:
            json.dump(args.__dict__, f, indent=4)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)

    # ==============================
    # Initialize Tensorboard
    # ==============================
    if coordinator.is_master():
        tensorboard_dir = f"{experiment_dir}/tensorboard"
        os.makedirs(tensorboard_dir, exist_ok=True)
        SummaryWriter(tensorboard_dir)

    # ==============================
    # Initialize Booster
    # ==============================
    if args.plugin == "gemini":
        plugin = GeminiPlugin(
            precision=args.mixed_precision,
            initial_scale=2**16,
            max_norm=args.grad_clip,
        )
    elif args.plugin == "gemini_auto":
        plugin = GeminiPlugin(
            precision=args.mixed_precision,
            placement_policy="auto",
            initial_scale=2**16,
            max_norm=args.grad_clip,
        )
    elif args.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=args.mixed_precision,
            initial_scale=2**16,
            max_norm=args.grad_clip,
        )
    elif args.plugin == "zero2_cpu":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=args.mixed_precision,
            initial_scale=2**16,
            cpu_offload=True,
            max_norm=args.grad_clip,
        )
    elif args.plugin == "3d":
        plugin = HybridParallelPlugin(
            tp_size=args.tp,
            pp_size=1,
            zero_stage=args.zero,
            max_norm=args.grad_clip,
            precision=args.mixed_precision,
        )
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")
    booster = Booster(plugin=plugin)

    # ======================================================
    # Initialize Model, Objective, Optimizer
    # ======================================================
    # Create model
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    dtype = torch.float16 if args.mixed_precision == "fp16" else torch.bfloat16
    model = DiT_models[args.model](input_size=latent_size, num_classes=args.num_classes).to(device).to(dtype)
    model_numel = get_model_numel(model)
    logger.info(f"Model params: {format_numel_str(model_numel)}")

    # Create ema and vae model
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    requires_grad(ema, False)
    diffusion = create_diffusion(timestep_respacing="")  # default: 1000 steps, linear noise schedule
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper)
    optimizer = HybridAdam(model.parameters(), lr=args.lr, weight_decay=0, adamw_mode=True)

    # Prepare models for training
    update_ema(ema, model, decay=0)  # Ensure EMA is initialized with synced weights
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Boost model for distributed training
    torch.set_default_dtype(dtype)
    model, optimizer, _, _, _ = booster.boost(model=model, optimizer=optimizer)
    torch.set_default_dtype(torch.float)

    # Setup data:
    # transform = transforms.Compose([
    #     transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    # ])
    # dataset = ImageFolder(args.data_path, transform=transform)
    # sampler = DistributedSampler(
    #     dataset,
    #     num_replicas=dist.get_world_size(),
    #     rank=rank,
    #     shuffle=True,
    #     seed=args.global_seed
    # )
    # loader = DataLoader(
    #     dataset,
    #     batch_size=int(args.global_batch_size // dist.get_world_size()),
    #     shuffle=False,
    #     sampler=sampler,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     drop_last=True
    # )
    # logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Variables for monitoring/logging purposes:
    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()

    batch = int(args.global_batch_size // dist.get_world_size())

    # TODO: load ckpt

    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        # for x, y in loader:
        for _ in tqdm.tqdm(range(100)):
            x = torch.randn(batch, 3, args.image_size, args.image_size).to(device)
            y = torch.randint(0, 1000, (batch,)).to(device)
            # x = x.to(device)
            # y = y.to(device)

            # VAE encode
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                x = x.to(dtype)

            # Diffusion
            t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
            model_kwargs = dict(y=y)
            loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
            loss = loss_dict["loss"].mean()
            booster.backward(loss=loss, optimizer=optimizer)
            optimizer.step()
            optimizer.zero_grad()

            # Update EMA
            update_ema(ema, model.module)

            # Log loss values:
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}"
                )
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time()

            # Save DiT checkpoint:
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "args": args,
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")
    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str)
    parser.add_argument(
        "--plugin", type=str, default="zero2", choices=["gemini", "gemini_auto", "zero2", "zero2_cpu", "3d"]
    )
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=16)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--lr", type=float, default=1e-4, help="Gradient clipping value")
    args = parser.parse_args()
    main(args)
