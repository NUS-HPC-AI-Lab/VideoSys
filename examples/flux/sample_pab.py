# process_captions.py
import os
import time
import argparse
import csv
from datetime import datetime
from typing import List, Tuple
from videosys.pipelines.flux.pipeline_flux_pab import FluxConfig, FluxPipeline, FluxPABConfig

def read_caption_file(caption_file: str, start_idx: int, end_idx: int) -> List[Tuple[str, str]]:
    """Read specific range of captions from file"""
    caption_data = []
    with open(caption_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= start_idx and i < end_idx:
                image_id, caption = line.strip().split(' ', 1)
                caption_data.append((image_id, caption))
            if i >= end_idx:
                break
    return caption_data

def log_to_csv(log_file: str, entries: List[dict]):
    """Write generation log to CSV file"""
    fieldnames = ['batch_id', 'image_ids', 'prompts', 'start_time', 'end_time', 'processing_time']
    
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(entries)

def process_captions(start_idx: int, end_idx: int, 
                    caption_file: str = 'FID_caption.txt',
                    output_dir: str = './outputs/flux-pab',
                    log_file: str = './logs/generation_log.csv',
                    batch_size: int = 4,
                    height: int = 1024,
                    width: int = 1024,
                    guidance_scale: float = 3.5,
                    num_steps: int = 50,
                    max_seq_len: int = 512):
    
    # Load captions
    caption_data = read_caption_file(caption_file, start_idx, end_idx)
    print(f"Processing captions from index {start_idx} to {end_idx}")
    
    # Configure pipeline
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
    
    # Initialize pipeline
    pipe = FluxPipeline(
        config=config,
        device="cuda"
    )
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Process captions
    log_entries = []
    total_batches = (len(caption_data) + batch_size - 1) // batch_size
    
    for batch_idx in range(total_batches):
        start_pos = batch_idx * batch_size
        end_pos = min(start_pos + batch_size, len(caption_data))
        batch_data = caption_data[start_pos:end_pos]
        batch_ids, batch_prompts = zip(*batch_data)
        
        start_time = datetime.now()
        
        # Generate images
        images = pipe(
            list(batch_prompts),
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            max_sequence_length=max_seq_len
        ).images
        
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Save images
        for image_id, image in zip(batch_ids, images):
            save_path = os.path.join(output_dir, f"{image_id}.png")
            image.save(save_path)
        
        # Log batch information
        log_entry = {
            'batch_id': batch_idx,
            'image_ids': ','.join(batch_ids),
            'prompts': '|'.join(batch_prompts),
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
            'processing_time': processing_time
        }
        log_entries.append(log_entry)
        
        print(f"Batch {batch_idx + 1}/{total_batches} processed in {processing_time:.2f}s")
        
        if len(log_entries) >= 5:
            log_to_csv(log_file, log_entries)
            log_entries = []
    
    if log_entries:
        log_to_csv(log_file, log_entries)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process captions with Flux pipeline')
    parser.add_argument('--start-idx', type=int, required=True)
    parser.add_argument('--end-idx', type=int, required=True)
    parser.add_argument('--caption-file', type=str, default='FID_caption.txt')
    parser.add_argument('--output-dir', type=str, default='./outputs/flux-pab')
    parser.add_argument('--log-file', type=str, default='./logs/generation_log.csv')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--height', type=int, default=1024)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--guidance-scale', type=float, default=3.5)
    parser.add_argument('--num-steps', type=int, default=50)
    parser.add_argument('--max-seq-len', type=int, default=512)
    
    args = parser.parse_args()
    process_captions(**vars(args))