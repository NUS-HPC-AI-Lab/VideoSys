import argparse
import os

import imageio
import json
import torch

from evaluations.fastvideodiffusion.eval.calculate_fvd import calculate_fvd
from evaluations.fastvideodiffusion.eval.calculate_lpips import calculate_lpips
from evaluations.fastvideodiffusion.eval.calculate_psnr import calculate_psnr
from evaluations.fastvideodiffusion.eval.calculate_ssim import calculate_ssim



def load_videos(directory, video_ids, file_extension):
    videos = []
    for video_id in video_ids:
        video_path = os.path.join(directory, f"{video_id}.{file_extension}")
        if os.path.exists(video_path):
            video = load_video(video_path)  # Define load_video based on how videos are stored
            videos.append(video)
        else:
            raise ValueError(f"Video {video_id}.{file_extension} not found in {directory}")
    return videos



def load_video(video_path):
    """
    Load a video from the given path and convert it to a PyTorch tensor.
    """
    # Read the video using imageio
    reader = imageio.get_reader(video_path, 'ffmpeg')
    
    # Extract frames and convert to a list of tensors
    frames = []
    for frame in reader:
        # Convert the frame to a tensor and permute the dimensions to match (C, H, W)
        frame_tensor = torch.tensor(frame).permute(2, 0, 1)
        frames.append(frame_tensor)
    
    # Stack the list of tensors into a single tensor with shape (T, C, H, W)
    video_tensor = torch.stack(frames)
    
    return video_tensor



def main(args):
    device = torch.device(f"cuda:{args.device}")
    
    eval_video_dir = args.eval_video_dataset
    generated_video_dir = args.generated_video_dataset

    video_ids = []
    for f in os.listdir(generated_video_dir):
        if f.endswith('.video'):
            video_ids.append(f.split('.')[0])
            file_extension = 'video'
        elif f.endswith('.mp4'):
            video_ids.append(f.split('.')[0])
            file_extension = 'mp4'
        else:
            raise ValueError(f"Unsupported file extension for video file: {f}")
                    
    if not video_ids:
        raise ValueError("No videos found in the generated video dataset. Exiting.")

    eval_videos = load_videos(eval_video_dir, video_ids, file_extension)
    
    generated_videos = load_videos(generated_video_dir, video_ids, file_extension)


    if len(eval_videos) == 0 or len(generated_videos) == 0:
        raise ValueError("No matching videos found in one or both directories. Exiting.")


    eval_videos_tensor = torch.stack(eval_videos).to(device)
    generated_videos_tensor = torch.stack(generated_videos).to(device)


    print(f"Loaded {len(eval_videos)} evaluation videos | {len(generated_videos)} generated videos")
    
    if args.calculate_lpips:
        result = calculate_lpips(eval_videos_tensor, generated_videos_tensor, device=device)
        
        print("LPIPS results:")
        print(json.dumps(result, indent=4))
        
    elif args.calculate_psnr:
        result = calculate_psnr(eval_videos_tensor, generated_videos_tensor)
        
        print("PSNR results:")
        print(json.dumps(result, indent=4))
        
    elif args.calculate_ssim:
        result = calculate_ssim(eval_videos_tensor, generated_videos_tensor)
        
        print("SSIM results:")
        print(json.dumps(result, indent=4))
        
    elif args.calculate_fvd:
        result = calculate_fvd(eval_videos_tensor, generated_videos_tensor, device=device, method= args.eval_method)
        
        print(f"FVD results with {args.eval_method}:")
        print(json.dumps(result, indent=4))




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # eval
    parser.add_argument("--calculate_fvd", action="store_true")
    parser.add_argument("--calculate_lpips", action="store_true")
    parser.add_argument("--calculate_psnr", action="store_true")
    parser.add_argument("--calculate_ssim", action="store_true")
    
    parser.add_argument("--eval_method", type=str, default="videogpt", required=True)
    
    # dataset
    parser.add_argument("--eval_video_dataset", type=str, default="./evaluations/fastvideodiffusion/datasets/webbvid", required=True)
    parser.add_argument("--generated_video_dataset", type=str, default="./evaluations/fastvideodiffusion/samples/latte", required=True)
    
    parser.add_argument("--device", type=str, default="0")
    
    args = parser.parse_args()

    main(args)