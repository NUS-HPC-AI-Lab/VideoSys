import argparse
import logging
import math
import os
import random
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
from videosys.models.transformers.open_sora_transformer_3d import STDiT3_XL_2, STDiT3Config
from videosys.schedulers.scheduling_rflow_open_sora import RFLOW
from videosys.training.ckpt_io import load, save, save_training_config
from videosys.training.datasets.open_sora.dataloader import prepare_dataloader
from videosys.training.datasets.open_sora.datasets import DummyVariableVideoTextDataset, VariableVideoTextDataset
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
        logging.info("mask ratios: %s", mask_ratios)
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


def get_text_embeddings(tokenizer, text_encoder, texts):
    text_tokens_and_mask = tokenizer(
        texts,
        max_length=300,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )
    device = text_encoder.device
    input_ids = text_tokens_and_mask["input_ids"].to(device)
    attention_mask = text_tokens_and_mask["attention_mask"].to(device)
    with torch.no_grad():
        text_encoder_embs = text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )["last_hidden_state"].detach()
    return text_encoder_embs, attention_mask


def encode_prompt(text_encoder, tokenizer, text):
    caption_embs, emb_masks = get_text_embeddings(tokenizer, text_encoder, text)
    caption_embs = caption_embs[:, None]
    emb_masks = None
    return dict(y=caption_embs, mask=emb_masks)


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
    init_logger(exp_dir)
    logging.info("Experiment directory created at %s", exp_dir)
    logging.info("Training configuration:\n %s", pformat(vars(args)))
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

    # TODO: scheduler is a better name?
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
        args.sp_balance_scope,
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
    logging.info("Building dataset...")
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
        dataset = VariableVideoTextDataset(
            transform_name="resize_crop", data_path=args.data_path, preprocessed_data=args.preprocessed_data
        )
    logging.info("Dataset contains %s samples.", len(dataset))

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
        sp_balance_scope=args.sp_balance_scope,
        auto_grad_accumulation=args.auto_grad_accumulation,
        max_grad_accumulation_steps=args.max_grad_accumulation_steps,
    )
    dataloader, sampler = prepare_dataloader(
        bucket_config=args.bucket_config,
        num_bucket_build_workers=args.num_bucket_build_workers,
        preprocessed_data=args.preprocessed_data,
        parallel_mgr=parallel_mgr,
        **dataloader_args,
    )

    # ======================================================
    # 3. build model
    # ======================================================
    logging.info("Building models...")

    # == build text-encoder and vae ==
    if not args.preprocessed_data:
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
        "[Diffusion] Trainable model params: %s, Total model params: %s",
        format_numel_str(model_numel_trainable),
        format_numel_str(model_numel),
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
        logging.info("Loaded checkpoint %s at epoch %s step %s", args.load, start_epoch, start_step)

    # == ema model sharding ==
    ema_sharding(model.module, ema)
    ema = ema.to(device, torch.float32)

    # == global variables ==
    cfg_epochs = args.epochs + (1 if profiler.need_profile() else 0)
    running_loss = 0.0
    logging.info("Training for %s epochs with profiling %s", args.epochs, profiler.need_profile())

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
        logging.info("Beginning %s...", epoch_desc)

        # == training loop in an epoch ==
        pbar = tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=epoch_desc,
            disable=not dist.get_rank() == 0,
            initial=start_step,
            total=num_steps_per_epoch,
        )
        for step, batch in pbar:
            # TODO: more elegant here
            profiler.optimize_dynamics(batch, model)

            total_gas = batch["gas"]
            iter_loss = 0.0
            for gas in range(total_gas):
                with profiler.profile(batch, model, gas) as valid_depth:
                    batch_data = batch["data"][gas]

                    if args.preprocessed_data:
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
                    exp_dir,
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
                    "Saved checkpoint at epoch %s, step %s, global_step %s to %s",
                    epoch,
                    step + 1,
                    global_step + 1,
                    save_dir,
                )

        if rank == 0 and not profiler.need_profile():
            logging.info(
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
    parser.add_argument("--sp-balance-scope", default="epoch", type=str, choices=["iter", "epoch"])
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
