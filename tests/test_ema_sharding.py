# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for DiT using PyTorch DDP.
"""
import argparse
import json
import os
import sys

sys.path.append("..")
from collections import OrderedDict
from copy import deepcopy
from glob import glob

import colossalai
import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.booster.plugin import GeminiPlugin, HybridParallelPlugin, LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device
from diffusers.models import AutoencoderKL
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from opendit.models.diffusion import create_diffusion
from opendit.models.dit import DiT_models
from opendit.utils.ckpt_utils import create_logger, load, record_model_param_shape, save
from opendit.utils.data_utils import center_crop_arr, prepare_dataloader
from opendit.utils.operation import model_sharding

# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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


def requires_grad(model: torch.nn.Module, flag: bool = True) -> None:
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


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
    device = get_current_device()

    # ==============================
    # Setup an experiment folder
    # ==============================
    # Make outputs folder (holds all experiment subfolders)
    os.makedirs(args.outputs, exist_ok=True)
    experiment_index = len(glob(f"{args.outputs}/*"))
    # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders)
    model_string_name = args.model.replace("/", "-")
    # Create an experiment folder
    experiment_dir = f"{args.outputs}/{experiment_index:03d}-{model_string_name}"
    if coordinator.is_master():
        os.makedirs(experiment_dir, exist_ok=True)
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
        writer = SummaryWriter(tensorboard_dir)

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
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()

    # Create ema and vae model
    # Note that parameter initialization is done within the DiT constructor
    # Create an EMA of the model for use after training
    ema = deepcopy(model).to(torch.float32).to(device)
    ema_unsharded = deepcopy(model).to(torch.float32).to(device)
    ema_backup = deepcopy(model).to(torch.float32).to(device)
    requires_grad(ema, False)
    ema_shape_dict = record_model_param_shape(ema)
    # default: 1000 steps, linear noise schedule
    diffusion = create_diffusion(timestep_respacing="")
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    # Setup optimizer
    # We used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper
    optimizer = HybridAdam(model.parameters(), lr=args.lr, weight_decay=0, adamw_mode=True)
    # You can use a lr scheduler if you want
    lr_scheduler = None

    # Prepare models for training
    # Ensure EMA is initialized with synced weights
    update_ema(ema, model, decay=0, sharded=False)
    update_ema(ema_unsharded, model, decay=0, sharded=False)
    # important! This enables embedding dropout for classifier-free guidance
    model.train()
    # EMA model should always be in eval mode
    ema.eval()

    # Setup data:
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )
    dataset = CIFAR10(args.data_path, transform=transform, download=True)
    dataloader = prepare_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Boost model for distributed training
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, dataloader=dataloader
    )

    torch.set_default_dtype(torch.float)

    # Variables for monitoring/logging purposes:
    start_epoch = 0
    start_step = 0
    sampler_start_idx = 0
    if args.load is not None:
        logger.info("Loading checkpoint")
        start_epoch, start_step, sampler_start_idx = load(booster, model, ema, optimizer, lr_scheduler, args.load)
        logger.info(f"Loaded checkpoint {args.load} at epoch {start_epoch} step {start_step}")
    model_sharding(ema)
    num_steps_per_epoch = len(dataloader)

    logger.info(f"Training for {args.epochs} epochs...")
    # if resume training, set the sampler start index to the correct value
    dataloader.sampler.set_start_index(sampler_start_idx)
    for epoch in range(start_epoch, args.epochs):
        dataloader.sampler.set_epoch(epoch)
        dataloader_iter = iter(dataloader)
        logger.info(f"Beginning epoch {epoch}...")
        with tqdm(
            range(start_step, num_steps_per_epoch),
            desc=f"Epoch {epoch}",
            disable=not coordinator.is_master(),
            total=num_steps_per_epoch,
            initial=start_step,
        ) as pbar:
            for step in pbar:
                x, y = next(dataloader_iter)
                x = x.to(device)
                y = y.to(device)

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
                update_ema(ema, model.module, optimizer=optimizer)
                update_ema(ema_unsharded, model.module, sharded=False)

                # Log loss values:
                all_reduce_mean(loss)
                if coordinator.is_master() and (step + 1) % args.log_every == 0:
                    pbar.set_postfix({"loss": loss.item()})
                    writer.add_scalar("loss", loss.item(), epoch * num_steps_per_epoch + step)

                if args.ckpt_every > 0 and (step + 1) % args.ckpt_every == 0:
                    logger.info(f"Saving checkpoint")
                    save(
                        booster,
                        model,
                        ema,
                        optimizer,
                        lr_scheduler,
                        epoch,
                        step + 1,
                        args.batch_size,
                        coordinator,
                        experiment_dir,
                        ema_shape_dict,
                    )
                    logger.info(f"Saved checkpoint at epoch {epoch} step {step + 1} to {experiment_dir}")
                    break

        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        dataloader.sampler.set_start_index(0)
        start_step = 0
    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    test_flag = True
    if coordinator.is_master():
        logger.info("Loading sharded ema model from {}".format(experiment_dir))
        # Load sharded ema parameters into the backup model for comparison
        ema_backup.load_state_dict(torch.load(os.path.join(experiment_dir, "epoch0-step10/ema.pt")))
        # Compare sharded and unsharded ema model parameters
        unsharded_param = OrderedDict(ema_unsharded.named_parameters())
        for name, param in OrderedDict(ema_backup.named_parameters()).items():
            if not torch.allclose(param, unsharded_param[name]):
                logger.warning(f"Sharded and unsharded EMA parameters differ for {name}")
                test_flag = False
        if test_flag:
            logger.info("Sharded and unsharded EMA parameters match")
    logger.info("Done!")


if __name__ == "__main__":
    # Default args here will train DiT-XL/2 with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="./datasets")
    parser.add_argument(
        "--plugin", type=str, default="zero2", choices=["gemini", "gemini_auto", "zero2", "zero2_cpu", "3d"]
    )
    parser.add_argument("--outputs", type=str, default="outputs")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=2)
    parser.add_argument("--ckpt-every", type=int, default=10)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--lr", type=float, default=1e-4, help="Gradient clipping value")
    parser.add_argument("--grad_checkpoint", action="store_true", help="Use gradient checkpointing")
    args = parser.parse_args()
    main(args)