import argparse
import logging
import os
from datetime import timedelta

import deepspeed
import pandas as pd
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel

from videosys.core.distributed.parallel_mgr import set_distributed_state
from videosys.models.autoencoders.autoencoder_kl_open_sora import OpenSoraVAE_V1_2
from videosys.training.datasets.open_sora.datasets import TextPreProcessDataset, VideoPreProcesssDataset
from videosys.utils.logging import init_logger
from videosys.utils.utils import merge_args, set_seed, str_to_dtype


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
    return dict(y=caption_embs, mask=emb_masks)


@torch.no_grad()
def main(args):
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == device and dtype ==
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    dtype = str_to_dtype(args.dtype)

    # == init distributed training ==
    # NOTE: A very large timeout is set to avoid some processes exit early
    rank, world_size, _, _ = set_distributed_state()
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
    torch.set_num_threads(1)

    # == init logger, tensorboard & wandb ==
    init_logger()

    # ======================================================
    # 2. build dataset and dataloader
    # ======================================================
    logging.info("Building dataset...")
    # == build dataset ==
    video_list, text_list = [], []
    video_dataset = VideoPreProcesssDataset(transform_name="resize_crop", data_path=args.data_path)
    text_dataset = TextPreProcessDataset(data_path=args.data_path)
    logging.info("Dataset contains %s samples.", len(video_dataset))

    # == build dataloader ==
    video_dataloader = DataLoader(
        video_dataset, batch_size=1, num_workers=args.num_workers, prefetch_factor=args.prefetch_factor, pin_memory=True
    )
    text_dataloader = DataLoader(text_dataset, batch_size=args.batch_size)

    # ======================================================
    # 3. build model
    # ======================================================
    logging.info("Building models...")

    # == build text-encoder and vae ==
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

    # =======================================================
    # 4. get emb
    # =======================================================

    # == vae encode ==
    dist.barrier()
    logging.info("Start vae encoding")
    os.makedirs(args.output_emb_path, exist_ok=True)
    for batch in tqdm(video_dataloader, total=len(video_dataloader)):
        x = batch["video"].to(device, dtype)
        x = vae.encode(x)
        for i in range(x.shape[0]):
            emb_path = os.path.join(
                args.output_emb_path, os.path.basename(batch["path"][i]) + f"_{int(batch['index'][i])}_vae.pt"
            )
            torch.save(x[i], emb_path)
            video_list.append(emb_path)

    # == text encode ==
    dist.barrier()
    logging.info("Start text encoding")
    os.makedirs(args.output_emb_path, exist_ok=True)
    for batch in tqdm(text_dataloader, total=len(text_dataloader)):
        y = batch["text"]
        model_args = encode_prompt(text_encoder, tokenizer, y)
        for i in range(len(y)):
            cur_model_args = {k: v[i] for k, v in model_args.items()}
            emb_path = os.path.join(
                args.output_emb_path, os.path.basename(batch["path"][i]) + f"_{int(batch['index'][i])}_text.pt"
            )
            torch.save(cur_model_args, emb_path)
            text_list.append(emb_path)

    # == save data_df ==
    dist.barrier()
    video_lists = [None for _ in range(dist.get_world_size())]
    text_lists = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(video_lists, video_list)
    dist.all_gather_object(text_lists, text_list)
    if dist.get_rank() == 0:
        video_list = [v for vl in video_lists for v in vl]
        text_list = [t for tl in text_lists for t in tl]
        data_df = pd.read_csv(args.data_path)
        data_df["vae_emb"] = video_list
        data_df["text_emb"] = text_list
        data_df.to_csv(args.output_csv_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # model config
    parser.add_argument("config", help="model config file path")

    # path
    parser.add_argument("--data-path", default="assets/example_data/demo.csv", type=str, help="path to input data csv")
    parser.add_argument(
        "--output-emb-path", default="assets/example_data/embs", type=str, help="outputs embedding path"
    )
    parser.add_argument(
        "--output-csv-path", default="assets/example_data/demo_preprocess.csv", type=str, help="outputs csv path"
    )

    # settings
    parser.add_argument("--seed", default=1024, type=int, help="seed for reproducibility")
    parser.add_argument("--batch-size", default=2, type=int, help="batch size")
    parser.add_argument("--dtype", default="bf16", type=str, help="data type")
    parser.add_argument("--num-workers", default=4, type=int, help="number of workers")
    parser.add_argument("--prefetch-factor", default=2, type=int, help="prefetch factor")

    args = parser.parse_args()
    config_args = OmegaConf.load(args.config)
    args = merge_args(args, config_args)

    main(args)
