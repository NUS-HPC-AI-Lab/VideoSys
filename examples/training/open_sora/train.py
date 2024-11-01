import argparse
import os
from copy import deepcopy
from datetime import timedelta
from pprint import pformat

import deepspeed
import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from videosys.core.dcp.profiler import Profiler, set_profiler
from videosys.core.distributed.parallel_mgr import DynamicParallelManager, ParallelManager, set_distributed_state
from videosys.models.transformers.open_sora_transformer_3d import STDiT3_XL_2, STDiT3Config
from videosys.schedulers.scheduling_rflow_open_sora import RFLOW
from videosys.training.ckpt_io.open_sora import (
    define_experiment_workspace,
    model_gathering,
    model_sharding,
    record_model_param_shape,
    save,
    save_training_config,
)
from videosys.training.datasets.open_sora.dataloader import prepare_dataloader
from videosys.training.datasets.open_sora.datasets import DummyVariableVideoTextDataset, VariableVideoTextDataset
from videosys.training.lr_schedulers.linear_warmup_open_sora import LinearWarmupLR
from videosys.utils.logging import logger
from videosys.utils.training import (
    MaskGenerator,
    all_reduce_mean,
    format_numel_str,
    get_model_numel,
    requires_grad,
    update_ema,
)
from videosys.utils.utils import merge_args, set_seed, str_to_dtype


def train_step(batch, model, mask_generator, scheduler, lr_scheduler, profiler: Profiler, device, dtype):
    profiler.optimize_dynamics(batch, model)

    total_gas = batch["gas"]
    iter_loss = 0.0
    for gas in range(total_gas):
        with profiler.profile(batch, model, gas) as valid_depth:
            batch_data = batch["data"][gas]

            # move data
            x = batch_data.pop("video").to(device, dtype)  # [B, C, T, H, W]
            y = batch_data.pop("text").to(device, dtype)
            mask = batch_data.pop("mask").to(device)
            model_args = dict(y=y, mask=mask)

            for k, v in batch_data.items():
                if isinstance(v, torch.Tensor):
                    model_args[k] = v.to(device, dtype)
            model_args["valid_depth"] = valid_depth

            # mask
            mask = None
            if mask_generator is not None:
                mask = mask_generator.get_masks(x)
                model_args["x_mask"] = mask

            # diffusion
            loss_dict = scheduler.training_losses(model, x, model_args, mask=mask)

            # backward
            profiler.set_gradient_accumulation_boundary(model, batch, gas)

            loss = loss_dict["loss"].mean()
            model.backward(loss)

            model.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            iter_loss += loss.detach()

    return iter_loss


def main(args):
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == device and dtype ==
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert args.dtype in ["fp16", "bf16"], f"Unknown mixed precision {args.dtype}"
    dtype = str_to_dtype(args.dtype)

    # == init distributed training ==
    # NOTE: A very large timeout is set to avoid some processes exit early
    rank, world_size, node_rank, node_size = set_distributed_state(args.distributed_profile)
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend="nccl",
        timeout=timedelta(hours=24),
    )
    deepspeed.init_distributed(timeout=timedelta(seconds=10))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    set_seed(args.seed)
    device = torch.cuda.current_device()

    # == init exp_dir ==
    exp_name, exp_dir = define_experiment_workspace(args.outputs)
    dist.barrier()
    if dist.get_rank() == 0:
        os.makedirs(exp_dir, exist_ok=True)
        save_training_config(vars(args), exp_dir)
    dist.barrier()

    # == init logger, tensorboard & wandb ==
    logger.info("Experiment directory created at %s", exp_dir)
    logger.info("Training configuration:\n %s", pformat(vars(args)))
    if dist.get_rank() == 0:
        if args.wandb:
            wandb.init(project="Open-Sora", name=exp_name, config=vars(args), dir="./outputs/wandb")

    # == init parallel manager ==
    if args.dynamic_sp:
        parallel_mgr = DynamicParallelManager(dist.get_world_size(), args.sampler_schedule_type == "local")
    else:
        parallel_mgr = ParallelManager(dist.get_world_size() // args.sp_size, 1, args.sp_size)

    torch.set_num_threads(1)

    # =======================================================
    # bonus: profile for better batching
    # =======================================================
    # TODO: hardcoded for T5
    text_max_seq_len = 300
    text_hidden_size = 4096
    model_config = STDiT3Config.from_pretrained(args.ckpt_path)

    profiler: Profiler = set_profiler(
        model_config,
        args.bucket_config,
        text_max_seq_len,
        text_hidden_size,
        device,
        dtype,
        args.dynamic_sp,
        args.dynamic_recompute,
        args.auto_grad_accumulation,
        args.profile,
        args.end2end_profile,
        args.distributed_profile,
        node_rank,
        node_size,
        args.alloc_memory_fraction,
        exp_dir,
        args.profile_path,
        parallel_mgr,
    )

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logger.info("Building dataset...")
    # == build dataset ==
    if args.dummy_dataset:
        dataset = DummyVariableVideoTextDataset(
            data_size=args.dummy_data_size,
            seed=args.seed,
            data_path=args.data_path,
            transform_name="resize_crop",
            preprocessed_data=args.preprocessed_data,
            bucket_config=args.bucket_config,
            distribution=args.distribution,
            zipf_offset=args.zipf_offset,
            image_mixing_type=args.image_mixing_type,
            image_mixing_frac=args.image_mixing_frac,
            res_scale=args.res_scale,
            frame_scale=args.frame_scale,
        )
    else:
        dataset = VariableVideoTextDataset(transform_name="resize_crop", data_path=args.data_path)
    logger.info("Dataset contains %s samples.", len(dataset))

    # == build dataloader ==
    dataloader_args = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        shuffle=True,
        drop_last=args.drop_last,
        keep_last=args.keep_last,
        pin_memory=True,
        process_group=parallel_mgr.dp_group,
        prefetch_factor=args.prefetch_factor,
        optimized_schedule=args.sampler_schedule_type if args.dynamic_sp else None,
        auto_grad_accumulation=args.auto_grad_accumulation,
        max_grad_accumulation_steps=args.max_grad_accumulation_steps,
    )
    dataloader, sampler = prepare_dataloader(
        bucket_config=args.bucket_config,
        num_bucket_build_workers=args.num_bucket_build_workers,
        preprocessed_data=args.preprocessed_data,
        **dataloader_args,
    )

    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")
    # == build text-encoder ==
    if args.preprocessed_data:
        text_encoder_output_dim = 4096
        text_encoder_model_max_length = 300
    # else:
    #     text_encoder = T5Encoder(
    #         from_pretrained="DeepFloyd/t5-v1_1-xxl", model_max_length=300, shardformer=True, device=device, dtype=dtype
    #     )
    #     text_encoder_output_dim = text_encoder.output_dim
    #     text_encoder_model_max_length = text_encoder.model_max_length

    # == build vae ==
    if args.preprocessed_data:
        latent_size = [None, None, None]
        vae_out_channels = 4
    # else:
    #     vae = OpenSoraVAE_V1_2(
    #         micro_frame_size=17,
    #         micro_batch_size=4,
    #     )
    #     vae = vae.to(device, dtype).eval()
    #     input_size = (dataset.num_frames, *dataset.image_size)
    #     latent_size = vae.get_latent_size(input_size)
    #     vae_out_channels = vae.out_channels

    # == build diffusion model ==
    model = (
        STDiT3_XL_2(
            from_pretrained=args.ckpt_path,
            qk_norm=True,
            enable_flash_attn=True,
            freeze_y_embedder=True,
            input_size=latent_size,
            in_channels=vae_out_channels,
            caption_channels=text_encoder_output_dim,
            model_max_length=text_encoder_model_max_length,
        )
        .to(device, dtype)
        .train()
    )
    model_numel, model_numel_trainable = get_model_numel(model)
    logger.info(
        "[Diffusion] Trainable model params: %s, Total model params: %s",
        format_numel_str(model_numel_trainable),
        format_numel_str(model_numel),
    )

    # == build ema for diffusion model ==
    ema = deepcopy(model).to(torch.float32).to(device)
    requires_grad(ema, False)
    ema_shape_dict = record_model_param_shape(ema)
    ema.eval()
    update_ema(ema, model, decay=0, sharded=False)

    # == setup loss function, build scheduler ==
    scheduler = RFLOW(
        use_timestep_transform=True,
        sample_method="logit-normal",
    )

    # == setup optimizer ==
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        # adamw_mode=True,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=args.adam_eps,
    )

    warmup_steps = args.warmup_steps

    if warmup_steps is None:
        lr_scheduler = None
    else:
        lr_scheduler = LinearWarmupLR(optimizer, warmup_steps=args.warmup_steps)

    # == additional preparation ==
    if args.grad_checkpoint:
        model.enable_grad_checkpointing()
    model.enable_parallel(parallel_mgr=parallel_mgr)

    if args.mask_ratios is not None:
        mask_generator = MaskGenerator(args.mask_ratios)

    # =======================================================
    # 4. distributed training preparation with colossalai
    # =======================================================
    logger.info("Preparing for distributed training...")
    # == boosting ==
    # NOTE: we set dtype first to make initialization of model consistent with the dtype; then reset it to the fp32 as we make diffusion scheduler in fp32
    torch.set_default_dtype(dtype)
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 1e8,  # dont print
        # "gradient_accumulation_steps": gradient_accumulation_steps,
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 1,
            "reduce_scatter": True,
            "allgather_bucket_size": 5e8,
            "reduce_bucket_size": 5e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "bf16": {"enabled": True},
    }
    # Initialize the model, optimizer, and lr scheduler
    model, optimizer, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config=ds_config,
    )

    torch.set_default_dtype(torch.float)
    logger.info("Boosting model for distributed training")

    start_epoch = start_step = log_step = acc_step = 0
    # TODO: resume functionality should consider the profiler status
    # == resume ==
    # if args.load is not None:
    #     logger.info("Loading checkpoint")
    #     ret = load(
    #         booster,
    #         args.load,
    #         model=model,
    #         ema=ema,
    #         optimizer=optimizer,
    #         lr_scheduler=lr_scheduler,
    #         sampler=None if args.start_from_scratch else sampler,
    #     )
    #     if not args.start_from_scratch:
    #         start_epoch, start_step = ret
    #     logger.info("Loaded checkpoint %s at epoch %s step %s", args.load, start_epoch, start_step)

    model_sharding(ema)

    # == global variables ==
    cfg_epochs = args.epochs + (1 if profiler.need_profile() else 0)
    running_loss = 0.0
    logger.info("Training for %s epochs with profiling %s", args.epochs, profiler.need_profile())

    # =======================================================
    # 5. training loop
    # =======================================================
    dist.barrier()

    profiler.register_timers(args.register_timer_keys)

    for epoch in range(start_epoch, cfg_epochs):
        if profiler.need_profile():
            num_steps_per_epoch = None
            dataloader_iter = profiler.get_data_iter()
            epoch_desc = "Profiling"
            profiler.init_profiler()
        else:
            # == set dataloader to new epoch ==
            sampler.set_epoch(epoch)
            num_steps_per_epoch = len(dataloader)
            dataloader_iter = iter(dataloader)
            epoch_desc = f"Epoch {epoch}"
        logger.info("Beginning %s...", epoch_desc)

        # == training loop in an epoch ==
        pbar = tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=epoch_desc,
            disable=not dist.get_rank() == 0,
            initial=start_step,
            total=num_steps_per_epoch,
        )
        for step, batch in pbar:
            iter_loss = train_step(batch, model, mask_generator, scheduler, lr_scheduler, profiler, device, dtype)

            if profiler.need_profile():
                continue

            # == update EMA ==
            # update_ema(ema, model.module, optimizer=optimizer, decay=args.ema_decay)

            # == update log info ==
            all_reduce_mean(iter_loss)
            running_loss += iter_loss.item()
            global_step = epoch * num_steps_per_epoch + step
            log_step += 1
            acc_step += 1

            # == logging ==
            if dist.get_rank() == 0 and (global_step + 1) % args.log_every == 0:
                avg_loss = running_loss / log_step
                # progress bar
                pbar.set_postfix({"loss": avg_loss, "step": step, "global_step": global_step})
                # wandb
                if args.wandb:
                    wandb.log(
                        {
                            "iter": global_step,
                            "acc_step": acc_step,
                            "epoch": epoch,
                            "loss": loss.item(),
                            "avg_loss": avg_loss,
                            "lr": optimizer.param_groups[0]["lr"],
                            "debug/move_data_time": move_data_t.elapsed_time,
                            "debug/encode_time": encode_t.elapsed_time,
                            "debug/mask_time": mask_t.elapsed_time,
                            "debug/diffusion_time": loss_t.elapsed_time,
                            "debug/backward_time": backward_t.elapsed_time,
                            "debug/update_ema_time": ema_t.elapsed_time,
                            "debug/reduce_loss_time": reduce_loss_t.elapsed_time,
                        },
                        step=global_step,
                    )

                running_loss = 0.0
                log_step = 0

            # == checkpoint saving ==
            ckpt_every = args.ckpt_every
            if ckpt_every > 0 and (global_step + 1) % ckpt_every == 0:
                model_gathering(ema, ema_shape_dict)
                save_dir = save(
                    booster,
                    exp_dir,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    sampler=sampler,
                    epoch=epoch,
                    step=step + 1,
                    global_step=global_step + 1,
                    batch_size=args.batch_size,
                )
                if dist.get_rank() == 0:
                    model_sharding(ema)
                logger.info(
                    "Saved checkpoint at epoch %s, step %s, global_step %s to %s",
                    epoch,
                    step + 1,
                    global_step + 1,
                    save_dir,
                )
            if len(profiler.registered_timer_keys) > 0:
                log_str = f"Rank {dist.get_rank()} | Epoch {epoch} | Step {step} | "
                log_str += profiler.registered_timer_log()
                print(log_str)

        if rank == 0 and not profiler.need_profile():
            logger.info(
                f"Epoch {epoch}: steps: {num_steps_per_epoch} effective samples: {sampler.effective_samples}, "
                f"throughput: {sampler.effective_samples / pbar.format_dict['elapsed']} samples/s"
            )

        sampler.reset()
        start_step = 0

    dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model config
    parser.add_argument("config", help="model config file path")

    parser.add_argument("--seed", default=1024, type=int, help="seed for reproducibility")
    parser.add_argument("--batch-size", default=None, type=int, help="batch size")
    parser.add_argument("--outputs", default="./outputs", type=str, help="the dir to save model weights")
    parser.add_argument("--data-path", default=None, type=str, help="path to data csv")
    parser.add_argument("--dtype", default="bf16", type=str, help="data type")
    parser.add_argument("--grad-clip", default=0, type=float, help="gradient clipping")
    parser.add_argument("--plugin", default="zero2", type=str, help="plugin")
    parser.add_argument("--sp-size", default=1, type=int, help="sequence parallelism size")
    parser.add_argument("--reduce-bucket-size-in-m", default=20, type=int, help="reduce bucket size in MB")
    parser.add_argument("--epochs", default=100, type=int, help="number of epochs")
    parser.add_argument("--num-workers", default=4, type=int, help="number of workers")
    parser.add_argument("--prefetch-factor", default=2, type=int, help="prefetch factor")
    parser.add_argument("--bucket-config", default=None, type=str, help="bucket config")
    parser.add_argument("--num-bucket-build-workers", default=1, type=int, help="number of bucket build workers")
    parser.add_argument("--weight-decay", default=0, type=float, help="weight decay")
    parser.add_argument("--adam-eps", default=1e-8, type=float, help="adam epsilon")
    parser.add_argument("--grad-checkpoint", default=False, action="store_true", help="gradient checkpoint")
    parser.add_argument("--mask-ratios", default=None, type=str, help="mask ratios")
    parser.add_argument("--ema-decay", default=0.99, type=float, help="ema decay")
    parser.add_argument("--log-every", default=1, type=int, help="log every")
    parser.add_argument("--ckpt-every", default=1000, type=int, help="checkpoint every")
    parser.add_argument("--ckpt-path", default="hpcai-tech/OpenSora-STDiT-v3", type=str, help="path to model ckpt")

    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--wandb", default=False, action="store_true", help="enable wandb")
    parser.add_argument("--load", default=None, type=str, help="path to continue training")
    parser.add_argument("--start-from-scratch", action="store_true", help="start training from scratch")
    parser.add_argument("--warmup-steps", default=None, type=int, help="warmup steps")

    parser.add_argument(
        "--register-timer-keys",
        default=[],
        type=str,
        nargs="+",
        help="register timer keys",
        choices=["move_data", "mask", "diffusion", "backward", "update_ema", "reduce_loss"],
    )

    # experimental features
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--keep-last", action="store_true")
    parser.add_argument("--dummy-dataset", action="store_true")
    parser.add_argument("--dummy-data-size", default=100, type=int)
    parser.add_argument("--preprocessed-data", action="store_true")
    parser.add_argument("--image-mixing-type", default="inclusive", type=str, choices=["inclusive", "exclusive"])
    parser.add_argument("--image-mixing-frac", default=-1.0, type=float)
    parser.add_argument("--distribution", default="uniform", type=str, choices=["zipf", "uniform"])
    parser.add_argument("--zipf-offset", type=int, default=5)
    parser.add_argument("--res-scale", default=None, type=float)
    parser.add_argument("--frame-scale", default=None, type=float)
    parser.add_argument("--dynamic-sp", action="store_true")
    parser.add_argument("--dynamic-recompute", action="store_true")
    parser.add_argument("--auto-grad-accumulation", action="store_true")
    parser.add_argument("--max-grad-accumulation-steps", default=2, type=int)
    parser.add_argument(
        "--alloc-memory-fraction",
        default=0.75,
        type=float,
        help="This is an empirical value (tuned on A100-SXM4-40GB) to cap the allocated memory during profiling with dynamic sp. Communication in different ranks can cause free memory discrepancy, which can leads to comm deadlock. So you need to leave enough space to bear this discrepancy. If you meet this problem during profiling, try to decrease this value.",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-path", default=None, type=str)
    parser.add_argument("--distributed-profile", action="store_true")

    # deprecated features
    parser.add_argument("--end2end-profile", action="store_true")
    parser.add_argument("--sampler-schedule-type", default="local", type=str)

    args = parser.parse_args()
    config_args = OmegaConf.load(args.config)
    args = merge_args(args, config_args)

    main(args)
