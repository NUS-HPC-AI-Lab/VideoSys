import argparse
import json
import os
from glob import glob

import colossalai
import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.cluster import DistCoordinator
from colossalai.nn.optimizer import HybridAdam
from colossalai.utils import get_current_device
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from opendit.core.comm import model_sharding
from opendit.core.parallel_mgr import get_parallel_manager, set_parallel_manager
from opendit.datasets.dataloader import prepare_dataloader
from opendit.embed.t5_text_emb import T5Encoder
from opendit.models.opensora.datasets import DatasetFromCSV, get_transforms_video
from opendit.models.opensora.scheduler import IDDPM
from opendit.models.opensora.stdit import STDiT_XL_2
from opendit.utils.ckpt_utils import create_logger, load, record_model_param_shape, save
from opendit.utils.train_utils import all_reduce_mean, format_numel_str, get_model_numel, requires_grad, update_ema
from opendit.utils.utils import str_to_dtype
from opendit.vae.wrapper import VideoAutoencoderKL

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
    dtype = str_to_dtype(args.mixed_precision)

    # ==============================
    # Setup an experiment folder
    # ==============================
    # Make outputs folder (holds all experiment subfolders)
    os.makedirs(args.outputs, exist_ok=True)
    experiment_index = len(glob(f"{args.outputs}/*"))
    # Create an experiment folder
    experiment_dir = f"{args.outputs}/{experiment_index:03d}-OpenSora"
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
    else:
        raise ValueError(f"Unknown plugin {args.plugin}")
    booster = Booster(plugin=plugin)

    # ==============================
    # Initialize Process Group
    # ==============================
    sp_size = args.sequence_parallel_size
    dp_size = dist.get_world_size() // sp_size
    set_parallel_manager(dp_size, sp_size, dp_axis=0, sp_axis=1)

    # ======================================================
    # Initialize Model, Objective, Optimizer
    # ======================================================
    # Create VAE encoder
    vae = VideoAutoencoderKL(args.vae_pretrained_path, split=4).to(device, dtype)

    # Configure input size
    input_size = (args.num_frames, args.image_size[0], args.image_size[1])
    latent_size = vae.get_latent_size(input_size)
    text_encoder = T5Encoder(
        args.text_pretrained_path, args.text_max_length, shardformer=args.text_speedup, device=device
    )

    # Shared model config for two models
    model_config = {
        "from_pretrained": args.model_pretrained_path,
        "time_scale": args.model_time_scale,
        "space_scale": args.model_space_scale,
        "input_size": latent_size,
        "in_channels": vae.out_channels,
        "caption_channels": text_encoder.output_dim,
        "model_max_length": text_encoder.model_max_length,
        "enable_layernorm_kernel": args.enable_layernorm_kernel,
    }

    # Create DiT model
    model = STDiT_XL_2(
        enable_flashattn=args.enable_flashattn,
        dtype=dtype,
        **model_config,
    ).to(device, dtype)

    model_numel = get_model_numel(model)
    logger.info(f"Model params: {format_numel_str(model_numel)}")
    if args.grad_checkpoint:
        model.enable_gradient_checkpointing()

    # Create ema and vae model
    # Note that parameter initialization is done within the DiT constructor
    # Create an EMA of the model for use after training
    ema = STDiT_XL_2(**model_config).to(device)
    ema = ema.to(torch.float32)
    ema.load_state_dict(model.state_dict())
    requires_grad(ema, False)
    ema_shape_dict = record_model_param_shape(ema)
    # Only shard ema model when using zero2 plugin
    shard_ema = True if args.plugin == "zero2" else False

    # Create diffusion
    # default: 1000 steps, linear noise schedule
    scheduler = IDDPM(timestep_respacing="")

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
    dataset = DatasetFromCSV(
        args.data_path,
        transform=get_transforms_video(args.image_size[0]),
        num_frames=args.num_frames,
        frame_interval=args.frame_interval,
    )
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

    # shard ema parameter
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
                batch = next(dataloader_iter)
                x = batch["video"].to(device)
                y = batch["text"]

                # VAE encode
                with torch.no_grad():
                    # Map input images to latent space + normalize latents:
                    x = vae.encode(x.to(dtype)).to(torch.float32)
                    # Prepare text inputs
                    model_args = text_encoder.encode(y)

                # Diffusion
                t = torch.randint(0, scheduler.num_timesteps, (x.shape[0],), device=device)
                loss_dict = scheduler.training_losses(model, x, t, model_args)

                # Backward & update
                loss = loss_dict["loss"].mean()
                booster.backward(loss=loss, optimizer=optimizer)
                optimizer.step()
                optimizer.zero_grad()

                # Update EMA
                update_ema(ema, model.module, optimizer=optimizer, sharded=shard_ema)

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

    # train
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5, help="Gradient clipping value")
    parser.add_argument("--grad_checkpoint", action="store_true", help="Use gradient checkpointing")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])

    parser.add_argument("--global_seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--ckpt_every", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping value")

    parser.add_argument("--outputs", type=str, default="./outputs", help="Path to the output directory")
    parser.add_argument("--load", type=str, default=None, help="Path to a checkpoint dir to load")
    parser.add_argument("--data_path", type=str, default="./datasets", help="Path to the dataset")

    # sample
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--image_size", nargs="+", type=int, default=[512, 512])
    parser.add_argument("--frame_interval", type=int, default=3)

    # model
    parser.add_argument("--model_space_scale", type=float, default=1.0)
    parser.add_argument("--model_time_scale", type=float, default=1.0)
    parser.add_argument("--model_pretrained_path", type=str, required=True)

    # vae
    parser.add_argument("--vae_pretrained_path", type=str, default="stabilityai/sd-vae-ft-ema")

    # text encoer
    parser.add_argument("--text_pretrained_path", type=str, default="t5-v1_1-xxl")
    parser.add_argument("--text_max_length", type=int, default=120)
    parser.add_argument("--text_speedup", action="store_true")

    # kernel
    parser.add_argument("--enable_layernorm_kernel", action="store_true", help="Enable apex layernorm kernel")
    parser.add_argument("--enable_flashattn", action="store_true", help="Enable flashattn kernel")

    # parallel
    parser.add_argument("--plugin", type=str, default="zero2")
    parser.add_argument("--sequence_parallel_size", type=int, default=1, help="Sequence parallel size, enable if > 1")

    args = parser.parse_args()
    main(args)
