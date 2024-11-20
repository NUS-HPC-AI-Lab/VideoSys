import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'

import time
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import torch
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from torch.utils.data import Dataset, DataLoader
import numpy as np
from videosys.pipelines.flux.pipeline_flux_pab import FluxConfig, FluxPipeline, FluxPABConfig
from torch.nn.parallel import DataParallel

class CaptionDataset(Dataset):
    """Dataset class for captions"""
    def __init__(self, caption_data: List[Tuple[str, str]]):
        self.caption_data = caption_data
    
    def __len__(self):
        return len(self.caption_data)
    
    def __getitem__(self, idx):
        return self.caption_data[idx]

def read_caption_file(caption_file: str) -> List[Tuple[str, str]]:
    """Read caption file and return list of (image_id, caption) tuples"""
    caption_data = []
    with open(caption_file, 'r') as f:
        for line in f:
            image_id, caption = line.strip().split(' ', 1)
            caption_data.append((image_id, caption))
    return caption_data

class ParallelFluxPipeline:
    def __init__(self, config):
        self.config = config
        
        # 初始化在主GPU(0)上
        self.device = torch.device("cuda:0")
        self.base_pipe = FluxPipeline(
            config=config,
            device=self.device,
        )
        
        # 使用DataParallel包装整个pipeline
        self.gpu_ids = list(range(torch.cuda.device_count()))
        self.base_pipe = DataParallel(self.base_pipe, device_ids=self.gpu_ids)

    def process_batch(self, caption_data: List[Tuple[str, str]], 
                     batch_size: int = 4, **generation_kwargs) -> None:
        """Process prompts in batches using multiple GPUs"""
        os.makedirs("./outputs/flux-pab", exist_ok=True)
        
        # 创建dataset和dataloader
        dataset = CaptionDataset(caption_data)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        total_batches = len(dataloader)
        
        for batch_idx, (batch_ids, batch_prompts) in enumerate(dataloader):
            start_time = time.time()
            
            # 生成图像
            images = self.base_pipe(
                list(batch_prompts),
                height=generation_kwargs.get('height', 1024),
                width=generation_kwargs.get('width', 1024),
                guidance_scale=generation_kwargs.get('guidance_scale', 3.5),
                num_inference_steps=generation_kwargs.get('num_inference_steps', 50),
                max_sequence_length=generation_kwargs.get('max_sequence_length', 512),
                generator=torch.Generator(self.device).manual_seed(batch_idx)
            ).images
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Batch {batch_idx + 1}/{total_batches} processed in {elapsed_time:.2f} s.")
            
            # 保存图像
            for image_id, prompt, image in zip(batch_ids, batch_prompts, images):
                save_path = f"./outputs/flux-pab/{image_id}.png"
                image.save(save_path)
                print(f"Saved image {image_id} for prompt: '{prompt}'")

def run_parallel_generation():
    # 加载caption数据
    caption_data = read_caption_file("FID_caption.txt")
    print(f"Loaded {len(caption_data)} captions from file")
    
    # 配置PAB和Flux
    pab_config = FluxPABConfig(
        spatial_broadcast=True,
        spatial_threshold=[100, 930],
        spatial_range=5,
        temporal_broadcast=False,
        cross_broadcast=True,
        cross_threshold=[100, 930],
        cross_range=5,
        mlp_broadcast=True
    )
    config = FluxConfig(
        enable_pab=True,
        pab_config=pab_config
    )
    
    # 初始化并行pipeline
    pipe = ParallelFluxPipeline(config=config)
    
    # 计算合适的batch size
    num_gpus = torch.cuda.device_count()
    base_batch_size = 4
    total_batch_size = base_batch_size * num_gpus
    print(f"Using {num_gpus} GPUs with total batch size: {total_batch_size}")
    
    # 处理caption生成图像
    generation_kwargs = {
        'height': 1024,
        'width': 1024,
        'guidance_scale': 3.5,
        'num_inference_steps': 50,
        'max_sequence_length': 512
    }
    
    pipe.process_batch(
        caption_data=caption_data,
        batch_size=total_batch_size,
        **generation_kwargs
    )

if __name__ == "__main__":
    run_parallel_generation()