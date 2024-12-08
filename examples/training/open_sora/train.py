import argparse
import logging
import os
import time
from copy import deepcopy
from datetime import timedelta
from pprint import pformat

import deepspeed
import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel

from videosys.core.dcp.profiler import Profiler, set_profiler
from videosys.core.distributed.parallel_mgr import DynamicParallelManager, ParallelManager, set_distributed_state
from videosys.models.autoencoders.autoencoder_kl_open_sora import OpenSoraVAE_V1_2
from videosys.models.transformers.open_sora_transformer_3d import STDiT3_XL_2
from videosys.schedulers.scheduling_rflow_open_sora import RFLOW
from videosys.training.ckpt_io import load, save, save_training_config
from videosys.training.datasets.open_sora.dataloader import prepare_dataloader
from videosys.training.datasets.open_sora.datasets import DummyVariableVideoTextDataset, VariableVideoTextDataset
from videosys.training.datasets.open_sora.utils import MaskGenerator, encode_prompt
from videosys.training.ema_distributed import ema_gathering, ema_sharding, update_ema
from videosys.training.lr_schedulers.linear_warmup_open_sora import LinearWarmupLR
from videosys.utils.logging import init_logger
from videosys.utils.training import (
    all_reduce_mean,
    define_experiment_workspace,
    format_numel_str,
    get_model_numel,
    requires_grad,
)
from videosys.utils.utils import merge_args, set_seed, str_to_dtype


def main(args):
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == device and dtype ==
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    assert args.dtype in ["fp16", "bf16"], f"Unknown mixed precision {args.dtype}"
    dtype = str_to_dtype(args.dtype)

    # == init distributed training ==
    rank, world_size, node_rank, node_size = set_distributed_state(args.distributed_profile)
    dist.init_process_group(
        rank=rank,
        world_size=world_size,
        backend="nccl",
        timeout=timedelta(minutes=10),
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
    init_logger(exp_dir)
    logging.info(f"Experiment directory created at {exp_dir}")
    logging.info(f"Training configuration:\n {pformat(vars(args))}")
    if dist.get_rank() == 0:
        if args.wandb:
            wandb.init(project="Open-Sora", name=exp_name, config=vars(args), dir="./outputs/wandb")

    # == init parallel manager ==
    torch.set_num_threads(1)
    if args.dynamic_sp:
        parallel_mgr = DynamicParallelManager()
    else:
        parallel_mgr = ParallelManager(dist.get_world_size() // args.sp_size, 1, args.sp_size)
    preprocessed_data = args.preprocessed_data
    if args.profile_path is None or not os.path.exists(args.profile_path):
        do_profile = True
        preprocessed_data = True
        logging.info(
            f"[ATTENTION!] Profile file is not found at `{args.profile_path}`! Profiling will be performed then exit."
        )
    else:
        do_profile = False

    # ======================================================
    # 2. build model
    # ======================================================
    logging.info("Building models...")

    # == build text-encoder and vae ==
    if not preprocessed_data:
        text_encoder = T5EncoderModel.from_pretrained("DeepFloyd/t5-v1_1-xxl", torch_dtype=dtype).to(device).eval()
        tokenizer = AutoTokenizer.from_pretrained("DeepFloyd/t5-v1_1-xxl")
        vae = (
            OpenSoraVAE_V1_2(
                from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
                micro_frame_size=17,
                micro_batch_size=4,
            )
            .to(device, dtype)
            .eval()
        )

    # == build diffusion model ==
    model = STDiT3_XL_2(from_pretrained=args.ckpt_path, enable_flash_attn=True, torch_dtype=dtype).to(device).train()
    model_numel, model_numel_trainable = get_model_numel(model)
    logging.info(
        f"[Diffusion] Trainable model params: {format_numel_str(model_numel_trainable)}, "
        f"Total model params: {format_numel_str(model_numel)}",
    )

    # == build ema for diffusion model ==
    ema = deepcopy(model)
    requires_grad(ema, False)
    ema.eval()

    # == setup loss function, build scheduler ==
    scheduler = RFLOW(
        use_timestep_transform=True,
        sample_method="logit-normal",
    )

    # == setup optimizer ==
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
        eps=args.adam_eps,
    )

    # == setup learning rate scheduler ==
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

    # ======================================================
    # 3. build dataset and dataloader
    # ======================================================
    logging.info("Building dataset...")
    # create dcp profiler
    # TODO: scheduler is a better name?
    profiler: Profiler = set_profiler(
        total_layers=model.config.depth,
        bucket_config=args.bucket_config,
        text_max_seq_len=model.config.model_max_length,
        text_hidden_size=model.config.caption_channels,
        global_interpolation=not args.no_global_interpolation,
        dynamic_sp=args.dynamic_sp,
        dynamic_recompute=args.dynamic_recompute,
        auto_grad_acc=args.auto_grad_accumulation,
        do_profile=do_profile,
        distributed_profile=args.distributed_profile,
        node_rank=node_rank,
        node_size=node_size,
        alloc_fraction=args.alloc_memory_fraction,
        profile_path=args.profile_path,
        parallel_mgr=parallel_mgr,
        verbose=args.verbose,
    )

    # == build dataset ==
    if args.dummy_dataset:
        dataset = DummyVariableVideoTextDataset(
            data_size=args.dummy_data_size,
            seed=args.seed,
            data_path=args.data_path,
            transform_name="resize_crop",
            preprocessed_data=preprocessed_data,
            bucket_config=args.bucket_config,
            common_ar=args.common_ar,
            distribution=args.distribution,
            zipf_offset=args.zipf_offset,
            image_mixing_type=args.image_mixing_type,
            image_mixing_frac=args.image_mixing_frac,
        )
    else:
        dataset = VariableVideoTextDataset(
            transform_name="resize_crop", data_path=args.data_path, preprocessed_data=preprocessed_data
        )
    logging.info(f"Dataset contains {len(dataset)} samples.")

    # == build dataloader ==
    dataloader, sampler = prepare_dataloader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        shuffle=True,
        drop_last=args.drop_last,
        process_group=parallel_mgr.dp_group,
        prefetch_factor=args.prefetch_factor,
        auto_grad_accumulation=args.auto_grad_accumulation,
        bucket_config=args.bucket_config,
        num_bucket_build_workers=args.num_bucket_build_workers,
        parallel_mgr=parallel_mgr,
        calculate_imbalance=args.calculate_imbalance,
        verbose=args.verbose,
        max_grad_accumulation_steps=args.max_grad_accumulation_steps,
        min_grad_accumulation_steps=args.min_grad_accumulation_steps,
    )

    # =======================================================
    # 4. distributed training preparation
    # =======================================================
    logging.info("Preparing for distributed training...")
    # == boosting ==
    # we set dtype first to make initialization of model consistent with the dtype
    # then reset it to the fp32 as we make diffusion scheduler in fp32
    torch.set_default_dtype(dtype)
    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "steps_per_print": 1e8,  # dont print
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
    logging.info("Boosting model for distributed training")
    profiler.register_modules(
        {
            "spatial": model.module.spatial_blocks,
            "temporal": model.module.temporal_blocks,
        }
    )

    start_epoch = start_step = log_step = acc_step = 0
    # TODO: resume functionality should consider the profiler status
    # == resume ==
    if args.load is not None:
        logging.info("Loading checkpoint")
        ret = load(
            args.load,
            model=model,
            ema=ema,
            sampler=None if args.start_from_scratch else sampler,
        )
        if not args.start_from_scratch:
            start_epoch, start_step = ret
        logging.info(f"Loaded checkpoint {args.load} at epoch {start_epoch} step {start_step}")

    # == ema model sharding ==
    ema_sharding(model.module, ema)
    ema = ema.to(device, torch.float32)

    # == global variables ==
    if do_profile:
        start_epoch, cfg_epochs = 0, 1
    else:
        cfg_epochs = args.epochs
    running_loss = 0.0
    logging.info(f"Training for {args.epochs} epochs{' with profiling' if profiler.need_profile() else ''}.")

    # =======================================================
    # 5. training loop
    # =======================================================
    dist.barrier()
    token_counter = torch.zeros((1,), dtype=torch.double, device=device)

    for epoch in range(start_epoch, cfg_epochs):
        local_token_counter = 0.0
        if profiler.need_profile():
            # TODO: add timer for profile
            disable = True
            num_steps_per_epoch = None
            dataloader_iter = profiler.get_data_iter()
            epoch_desc = "Profiling"
            profiler.init_profiler()
        else:
            # == set dataloader to new epoch ==
            sampler.set_epoch(epoch)
            disable = not dist.get_rank() == 0
            num_steps_per_epoch = len(dataloader)
            dataloader_iter = iter(dataloader)
            epoch_desc = f"Epoch {epoch}"
        logging.info(f"Beginning {epoch_desc}...")

        # == training loop in an epoch ==
        pbar = tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=epoch_desc,
            disable=disable,
            initial=start_step,
            total=num_steps_per_epoch,
        )
        for step, batch in pbar:
            # TODO: more elegant here
            profiler.optimize_dynamics(batch, model)

            total_gas = batch["gas"]
            iter_loss = 0.0
            torch.cuda.synchronize()
            iter_start_time = time.time()
            for gas in range(total_gas):
                with profiler.profile(batch, model, gas) as valid_depth:
                    batch_data = batch["data"][gas]

                    if preprocessed_data:
                        # move data
                        x = batch_data.pop("video").to(device, dtype)  # [B, C, T, H, W]
                        y = batch_data.pop("text").to(device, dtype)
                        mask = batch_data.pop("mask").to(device)
                        model_args = dict(y=y, mask=mask)
                    else:
                        with torch.no_grad():
                            x = batch_data.pop("video").to(device, dtype)  # [B, C, T, H, W]
                            y = batch_data.pop("text")
                            # Prepare visual inputs
                            x = vae.encode(x)  # [B, C, T, H/P, W/P]
                            # Prepare text inputs
                            model_args = encode_prompt(text_encoder, tokenizer, y)

                    local_token_counter += x.shape[0] * x.shape[2] * x.shape[3] * x.shape[4] / parallel_mgr.sp_size

                    for k, v in batch_data.items():
                        if isinstance(v, torch.Tensor):
                            model_args[k] = v.to(device, dtype)
                    # TODO: polish
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

            if profiler.need_profile():
                continue

            # == update EMA ==
            update_ema(ema, model.module, decay=args.ema_decay)

            # == update log info ==
            all_reduce_mean(iter_loss)
            iter_loss = iter_loss.item() / total_gas
            running_loss += iter_loss
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
                            "loss": iter_loss,
                            "avg_loss": avg_loss,
                            "lr": optimizer.param_groups[0]["lr"],
                        },
                        step=global_step,
                    )

                running_loss = 0.0
                log_step = 0

            # == checkpoint saving ==
            if args.ckpt_every > 0 and (global_step + 1) % args.ckpt_every == 0:
                ema_gathering(model.module, ema)
                save_dir = save(
                    save_dir=exp_dir,
                    save_optimizer=args.save_optimizer,
                    model=model,
                    ema=ema,
                    sampler=sampler,
                    epoch=epoch,
                    step=step + 1,
                    global_step=global_step + 1,
                    batch_size=args.batch_size,
                )
                ema_sharding(model.module, ema)
                logging.info(
                    f"Saved checkpoint at epoch {epoch}, step {step + 1}, global_step {global_step + 1} to {save_dir}"
                )

            torch.cuda.synchronize()
            iter_elapsed_time = time.time() - iter_start_time
            logging.info(f"Iter: {step} / {epoch} elapsed: {iter_elapsed_time:.2f} s")

        token_counter.fill_(local_token_counter)
        dist.all_reduce(token_counter)
        if rank == 0 and not disable:
            elapsed_time = pbar.format_dict['elapsed']
            logging.info(
                f"Epoch {epoch}: steps: {num_steps_per_epoch} elapsed time: {elapsed_time:.2f} s"
                f", effective samples: {sampler.effective_samples}"
                f", sample throughput: {sampler.effective_samples / elapsed_time:.2f} samples/s"
                f", token throughput: {token_counter.item()/elapsed_time:.2f} token/s"
            )

        sampler.reset()
        start_step = 0
    dist.barrier()

    if do_profile:
        logging.info(
            f"Profiling is done and saved to {args.profile_path}. Please restart this programe for training with "
            f"`profile_path: {args.profile_path}` in the config file. Exiting..."
        )
    else:
        logging.info("Training is done. Exiting...")


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
    parser.add_argument("--ckpt-every", default=-1, type=int, help="checkpoint every")
    parser.add_argument("--ckpt-path", default="hpcai-tech/OpenSora-STDiT-v3", type=str, help="path to model ckpt")

    parser.add_argument("--lr", default=1e-4, type=float, help="learning rate")
    parser.add_argument("--wandb", default=False, action="store_true", help="enable wandb")
    parser.add_argument("--load", default=None, type=str, help="path to continue training")
    parser.add_argument("--start-from-scratch", action="store_true", help="start training from scratch")
    parser.add_argument("--warmup-steps", default=None, type=int, help="warmup steps")
    parser.add_argument("--verbose", action="store_true", help="verbose")
    parser.add_argument("--save-optimizer", action="store_true", help="save optimizer")

    # experimental features
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--dummy-dataset", action="store_true")
    parser.add_argument("--dummy-data-size", default=100, type=int)
    parser.add_argument("--common-ar", type=dict, default=None)
    parser.add_argument("--preprocessed-data", action="store_true")
    parser.add_argument("--image-mixing-type", default="exclusive", type=str, choices=["inclusive", "exclusive"])
    parser.add_argument("--image-mixing-frac", default=1, type=float)
    parser.add_argument("--distribution", default="zipf", type=str, choices=["zipf", "uniform"])
    parser.add_argument("--zipf-offset", type=int, default=5)
    parser.add_argument("--no-global-interpolation", action="store_true")
    parser.add_argument("--dynamic-sp", action="store_true")
    parser.add_argument("--dynamic-recompute", action="store_true")
    parser.add_argument("--auto-grad-accumulation", action="store_true")
    parser.add_argument(
        "--alloc-memory-fraction",
        default=0.70,
        type=float,
        help="This is an empirical value to cap the allocated memory during profiling with dynamic sp. Communication in different ranks can cause free memory discrepancy, which can leads to comm deadlock. So you need to leave enough space to bear this discrepancy. If you meet this problem during profiling, try to decrease this value.",
    )
    parser.add_argument("--profile-path", default="exp/profile", type=str)
    parser.add_argument("--distributed-profile", action="store_true")
    parser.add_argument("--calculate-imbalance", action="store_true")
    parser.add_argument("--max-grad-accumulation-steps", default=3, type=int)
    parser.add_argument("--min-grad-accumulation-steps", default=2, type=int)

    args = parser.parse_args()
    config_args = OmegaConf.load(args.config)
    args = merge_args(args, config_args)

    main(args)
