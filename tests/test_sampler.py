import argparse
import logging
import os
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
from videosys.models.transformers.open_sora_transformer_3d import STDiT3Config
from videosys.training.ckpt_io import save_training_config
from videosys.training.datasets.open_sora.dataloader import prepare_dataloader
from videosys.training.datasets.open_sora.datasets import DummyVariableVideoTextDataset, VariableVideoTextDataset
from videosys.utils.logging import init_logger
from videosys.utils.training import (
    define_experiment_workspace,
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
    logging.info(f"Experiment directory created at {exp_dir}")
    logging.info(f"Training configuration:\n {pformat(vars(args))}")
    if dist.get_rank() == 0:
        if args.wandb:
            wandb.init(project="Open-Sora", name=exp_name, config=vars(args), dir="./outputs/wandb")

    # == init parallel manager ==
    if args.dynamic_sp:
        parallel_mgr = DynamicParallelManager()
    else:
        parallel_mgr = ParallelManager(dist.get_world_size() // args.sp_size, 1, args.sp_size)

    torch.set_num_threads(1)

    # =======================================================
    # bonus: profile for better batching
    # =======================================================
    preprocessed_data = args.preprocessed_data
    model_config = STDiT3Config.from_pretrained(args.ckpt_path)
    if args.profile_path is None or not os.path.exists(args.profile_path):
        do_profile = True
        preprocessed_data = True
        logging.info(
            f"[ATTENTION!] Profile file is not found at `{args.profile_path}`! Profiling will be performed then exit."
        )
    else:
        do_profile = False
    # TODO: scheduler is a better name?
    profiler: Profiler = set_profiler(
        model_config=model_config,
        total_layers=model_config.depth,
        bucket_config=args.bucket_config,
        text_max_seq_len=model_config.model_max_length,
        text_hidden_size=model_config.caption_channels,
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
            preprocessed_data=preprocessed_data,
            bucket_config=args.bucket_config,
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
    )

    # == global variables ==
    start_epoch = 0
    start_step = 0
    cfg_epochs = 1

    # =======================================================
    # 5. training loop
    # =======================================================
    dist.barrier()

    def run_iteration(batch):
        parallel_mgr.set_sp_size(batch["sp_size"])
        assert batch["sp_size"] == parallel_mgr.sp_size

        total_gas = batch["gas"]
        iter_samples = torch.zeros(1, device=device, dtype=torch.float)

        for gas in range(total_gas):
            batch_data = batch["data"][gas]
            bs = batch_data["video"].shape[0]

            iter_samples += bs / batch["sp_size"]

        # ar_name = batch['ar_name']
        # num_frame = batch['num_frame']
        # h = batch["data"][0]["height"][0].item()
        # w = batch["data"][0]["width"][0].item()
        # bs = list(batch["data"][0]["video"].shape)
        # print(
        #     f">>> rank {rank} step: {step} gas: {gas}/{total_gas} sp: {get_sequence_parallel_rank()}/{get_sequence_parallel_size()}"
        #     f" ar: {ar_name} frame: {num_frame} h: {h} w: {w}, bs: {bs}"
        # )
        return iter_samples

    for epoch in range(start_epoch, cfg_epochs):
        # == set dataloader to new epoch ==
        sampler.set_epoch(epoch)
        num_steps_per_epoch = len(dataloader)
        dataloader_iter = iter(dataloader)
        logging.info("Beginning epoch %s...", epoch)

        logging.info(f"Epoch {epoch} has {num_steps_per_epoch} steps, {sampler.effective_samples} samples")
        acc_samples = torch.zeros(1, device=device, dtype=torch.float)

        # == training loop in an epoch ==
        with tqdm(
            enumerate(dataloader_iter, start=start_step),
            desc=f"Epoch {epoch}",
            disable=not dist.get_rank() == 0,
            initial=start_step,
            total=num_steps_per_epoch,
        ) as pbar:
            for step, batch in pbar:
                iter_samples = run_iteration(batch)
                
                dist.all_reduce(iter_samples)
                acc_samples += iter_samples
                global_step = epoch * num_steps_per_epoch + step

                # == logging ==
                if dist.get_rank() == 0 and (global_step + 1) % args.log_every == 0:
                    # progress bar
                    pbar.set_postfix(
                        {
                            "step": step,
                            "global_step": global_step,
                            # "acc_samples": int(acc_samples.item()),
                            "target": sampler.effective_samples,
                            # loss.item() should implicitly synchronize gpu&cpu in this process
                            "throughput": sampler.effective_samples / pbar.format_dict["elapsed"],
                        }
                    )

            if dist.get_rank() == 0:
                logging.info(
                    f"Epoch {epoch} has {num_steps_per_epoch} steps, {sampler.effective_samples} samples"
                    f", throughput: {sampler.effective_samples / pbar.format_dict['elapsed']}"
                    # f", final: {acc_samples.item()}"
                )
            assert (
                int(acc_samples.item()) == sampler.effective_samples
            ), f"runtime acc: {acc_samples.item()}, target: {sampler.effective_samples}"

        sampler.reset()
        start_step = 0


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

    # experimental features
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--dummy-dataset", action="store_true")
    parser.add_argument("--dummy-data-size", default=100, type=int)
    parser.add_argument("--preprocessed-data", action="store_true")
    parser.add_argument("--image-mixing-type", default="exclusive", type=str, choices=["inclusive", "exclusive"])
    parser.add_argument("--image-mixing-frac", default=-1.0, type=float)
    parser.add_argument("--distribution", default="zipf", type=str, choices=["zipf", "uniform"])
    parser.add_argument("--zipf-offset", type=int, default=5)
    parser.add_argument("--dynamic-sp", action="store_true")
    parser.add_argument("--dynamic-recompute", action="store_true")
    parser.add_argument("--auto-grad-accumulation", action="store_true")
    parser.add_argument(
        "--alloc-memory-fraction",
        default=0.75,
        type=float,
        help="This is an empirical value to cap the allocated memory during profiling with dynamic sp. Communication in different ranks can cause free memory discrepancy, which can leads to comm deadlock. So you need to leave enough space to bear this discrepancy. If you meet this problem during profiling, try to decrease this value.",
    )
    parser.add_argument("--profile-path", default="exp/profile", type=str)
    parser.add_argument("--distributed-profile", action="store_true")

    args = parser.parse_args()
    config_args = OmegaConf.load(args.config)
    args = merge_args(args, config_args)

    main(args)
