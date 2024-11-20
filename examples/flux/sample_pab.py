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
    fieldnames = ['batch_id', 'image_ids', 'prompts', 'start_time', 'end_time', 'processing_time', 'gpu_id']
    
    file_exists = os.path.isfile(log_file)
    with open(log_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(entries)

def process_captions(args):
    # Set GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    
    # Load captions for this GPU
    caption_data = read_caption_file(args.caption_file, args.start_idx, args.end_idx)
    print(f"GPU {args.gpu_id}: Processing captions from index {args.start_idx} to {args.end_idx}")
    
    # Configure PAB and Flux
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
        device=f"cuda:0"
    )
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    
    # Process captions in batches
    log_entries = []
    total_batches = (len(caption_data) + args.batch_size - 1) // args.batch_size
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(caption_data))
        batch_data = caption_data[start_idx:end_idx]
        batch_ids, batch_prompts = zip(*batch_data)
        
        # Record batch start time
        start_time = datetime.now()
        
        # Generate images for the batch
        images = pipe(
            list(batch_prompts),
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            max_sequence_length=args.max_seq_len
        ).images
        
        # Record batch end time
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        
        # Save images
        for image_id, image in zip(batch_ids, images):
            save_path = os.path.join(args.output_dir, f"{image_id}.png")
            image.save(save_path)
        
        # Create log entry for this batch
        log_entry = {
            'batch_id': batch_idx,
            'image_ids': ','.join(batch_ids),
            'prompts': '|'.join(batch_prompts),  # 使用|作为分隔符，因为prompt中可能包含逗号
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S.%f'),
            'processing_time': processing_time,
            'gpu_id': args.gpu_id
        }
        log_entries.append(log_entry)
        
        print(f"GPU {args.gpu_id}: Batch {batch_idx + 1}/{total_batches} processed in {processing_time:.2f}s")
        
        # Write to CSV every few batches
        if len(log_entries) >= 5:  # 每5个batch写入一次
            log_to_csv(args.log_file, log_entries)
            log_entries = []
    
    # Write any remaining log entries
    if log_entries:
        log_to_csv(args.log_file, log_entries)

def parse_args():
    parser = argparse.ArgumentParser(description='Process captions with Flux pipeline')
    
    # Required arguments
    parser.add_argument('--gpu-id', type=int, required=True,
                      help='GPU ID to use')
    parser.add_argument('--start-idx', type=int, required=True,
                      help='Start index of captions to process')
    parser.add_argument('--end-idx', type=int, required=True,
                      help='End index of captions to process')
    
    # Optional arguments with defaults
    parser.add_argument('--caption-file', type=str, default='FID_caption.txt',
                      help='Path to caption file')
    parser.add_argument('--output-dir', type=str, default='./outputs/flux-pab',
                      help='Output directory for generated images')
    parser.add_argument('--log-file', type=str, default='./logs/generation_log.csv',
                      help='Path to CSV log file')
    parser.add_argument('--batch-size', type=int, default=4,
                      help='Batch size for processing')
    parser.add_argument('--height', type=int, default=1024,
                      help='Height of generated images')
    parser.add_argument('--width', type=int, default=1024,
                      help='Width of generated images')
    parser.add_argument('--guidance-scale', type=float, default=3.5,
                      help='Guidance scale for generation')
    parser.add_argument('--num-steps', type=int, default=50,
                      help='Number of inference steps')
    parser.add_argument('--max-seq-len', type=int, default=512,
                      help='Maximum sequence length')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    process_captions(args)