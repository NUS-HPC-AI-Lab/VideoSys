import argparse
import json
import os
from glob import glob

import colossalai
import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin, TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device
from diffusers.models import AutoencoderKL
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from opendit.core.comm import model_sharding
from opendit.core.parallel_mgr import get_parallel_manager, set_parallel_manager
from opendit.datasets.dataloader import prepare_dataloader
from opendit.datasets.image_transform import get_transforms_image
from opendit.diffusion import create_diffusion
from opendit.models.dit import DiT, DiT_models
from opendit.utils.ckpt_utils import create_logger, load, record_model_param_shape, save
from opendit.utils.train_utils import all_reduce_mean, format_numel_str, get_model_numel, requires_grad, update_ema

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # ==============================
    # Initialize Distributed Training
    # ==============================
    colossalai.launch_from_torch({}, seed=args.global_seed)
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
    dist.barrier()
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
    if args.plugin == "zero2":
        plugin = LowLevelZeroPlugin(
            stage=2,
            precision=args.mixed_precision,
            initial_scale=2**16,
            max_norm=args.grad_clip,
        )
    elif args.plugin == "ddp":
        plugin = TorchDDPPlugin()
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")
    booster = Booster(plugin=plugin)

    # ==============================
    # Initialize Process Group
    # ==============================
    sp_size = 1  # image doesn't need sequence parallel
    dp_size = dist.get_world_size() // sp_size
    set_parallel_manager(dp_size, sp_size, dp_axis=0, sp_axis=1)

    # ======================================================
    # Initialize Model, Objective, Optimizer
    # ======================================================
    # Set mixed precision
    if args.mixed_precision == "bf16" and args.plugin != "ddp":
        dtype = torch.bfloat16
    elif args.mixed_precision == "fp16" and args.plugin != "ddp":
        dtype = torch.float16
    elif args.mixed_precision == "fp32" and args.plugin == "ddp":
        dtype = torch.float32
    else:
        raise ValueError(f"Unknown mixed precision {args.mixed_precision}")

    # Create VAE encoder
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device).to(dtype)

    # Configure input size
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    input_size = args.image_size // 8

    # Shared model config for two models
    model_config = {
        "input_size": input_size,
        "num_classes": args.num_classes,
        "enable_layernorm_kernel": args.enable_layernorm_kernel,
        "enable_modulate_kernel": args.enable_modulate_kernel,
    }

    # Create DiT model
    model: DiT = (
        DiT_models[args.model](
            enable_flashattn=args.enable_flashattn,
            dtype=dtype,
            **model_config,
        )
        .to(device)
        .to(dtype)
    )

    model_numel = get_model_numel(model)
    logger.info(f"Model params: {format_numel_str(model_numel)}")
    if args.grad_checkpoint:
        model.enable_gradient_checkpointing()

    # Create ema and vae model
    # Note that parameter initialization is done within the DiT constructor
    # Create an EMA of the model for use after training
    ema = DiT_models[args.model](**model_config).to(device)
    ema = ema.to(torch.float32)
    ema.load_state_dict(model.state_dict())
    requires_grad(ema, False)
    ema_shape_dict = record_model_param_shape(ema)

    # Create diffusion
    # default: 1000 steps, linear noise schedule
    diffusion = create_diffusion(timestep_respacing="")

    # Setup optimizer
    # We used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper
    optimizer = HybridAdam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0, adamw_mode=True
    )
    # You can use a lr scheduler if you want
    # Recommend if you continue training from a model
    lr_scheduler = None

    # Prepare models for training
    # Ensure EMA is initialized with synced weights
    update_ema(ema, model, decay=0, sharded=False)
    # important! This enables embedding dropout for classifier-free guidance
    model.train()
    # EMA model should always be in eval mode
    ema.eval()

    # Setup data:
    # master process goes first
    if not coordinator.is_master():
        dist.barrier()
    # To use ImageNet, you need to download it and:
    # from torchvision.datasets import ImageFolder
    # dataset = ImageFolder(args.data_path, transform=get_transforms_image(args.image_size))
    dataset = CIFAR10(args.data_path, transform=get_transforms_image(args.image_size), download=True)
    if coordinator.is_master():
        dist.barrier()
    dataloader = prepare_dataloader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
        pg_manager=get_parallel_manager(),
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")

    # Boost model for distributed training
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler, dataloader=dataloader
    )
    torch.set_default_dtype(torch.float)
    logger.info("Boost model for distributed training")

    # Variables for monitoring/logging purposes:
    start_epoch = 0
    start_step = 0
    sampler_start_idx = 0
    if args.load is not None:
        logger.info("Loading checkpoint")
        start_epoch, start_step, sampler_start_idx = load(booster, model, ema, optimizer, lr_scheduler, args.load)
        logger.info(f"Loaded checkpoint {args.load} at epoch {start_epoch} step {start_step}")

    # Only shard ema model when using zero2 plugin
    shard_ema = True if args.plugin == "zero2" else False
    if shard_ema:
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
                    x = x.to(dtype)
                    x = vae.encode(x).latent_dist.sample().mul_(0.18215)
                    # cast back to fp32 for bettet diffusion accuracy
                    x = x.to(torch.float32)

                # Diffusion
                t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
                model_kwargs = dict(y=y)
                loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
                loss = loss_dict["loss"].mean()
                booster.backward(loss=loss, optimizer=optimizer)
                optimizer.step()
                optimizer.zero_grad()

                # Update EMA
                update_ema(ema, model.unwrap(), optimizer=optimizer, sharded=shard_ema)

                # Log loss values:
                all_reduce_mean(loss)
                global_step = epoch * num_steps_per_epoch + step
                pbar.set_postfix({"loss": loss.item(), "step": step, "global_step": global_step})

                # Log to tensorboard
                if coordinator.is_master() and (global_step + 1) % args.log_every == 0:
                    writer.add_scalar("loss", loss.item(), global_step)

                # Save checkpoint
                if args.ckpt_every > 0 and (global_step + 1) % args.ckpt_every == 0:
                    logger.info(f"Saving checkpoint...")
                    save(
                        booster,
                        model,
                        ema,
                        optimizer,
                        lr_scheduler,
                        epoch,
                        step + 1,
                        global_step + 1,
                        args.batch_size,
                        coordinator,
                        experiment_dir,
                        ema_shape_dict,
                        shard_ema,
                    )
                    logger.info(
                        f"Saved checkpoint at epoch {epoch} step {step + 1} global_step {global_step + 1} to {experiment_dir}"
                    )

        # the continue epochs are not resumed, so we need to reset the sampler start index and start step
        dataloader.sampler.set_start_index(0)
        start_step = 0

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=DiT_models.keys(), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--plugin", type=str, default="zero2")
    parser.add_argument("--outputs", type=str, default="./outputs", help="Path to the output directory")
    parser.add_argument("--load", type=str, default=None, help="Path to a checkpoint dir to load")

    parser.add_argument("--data_path", type=str, default="./datasets", help="Path to the dataset")
    parser.add_argument("--image_size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num_classes", type=int, default=1000)

    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=1000)

    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--lr", type=float, default=1e-4, help="Gradient clipping value")
    parser.add_argument("--grad_checkpoint", action="store_true", help="Use gradient checkpointing")

    parser.add_argument("--enable_modulate_kernel", action="store_true", help="Enable triton modulate kernel")
    parser.add_argument("--enable_layernorm_kernel", action="store_true", help="Enable apex layernorm kernel")
    parser.add_argument("--enable_flashattn", action="store_true", help="Enable flashattn kernel")

    args = parser.parse_args()
    main(args)
