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


def preprocess_eval_videos(eval_videos, generated_video_shape):
    T_gen, C_gen, H_gen, W_gen = generated_video_shape
    preprocessed_videos = []
    
    for video in eval_videos:
        T_eval, C_eval, H_eval, W_eval = video.shape
        
        if T_eval < T_gen:
            raise ValueError(f"Eval video time steps ({T_eval}) are less than generated video time steps ({T_gen}).")
        
        if H_eval < H_gen or W_eval < W_gen:
            raise ValueError(f"Eval video dimensions ({H_eval}x{W_eval}) are less than generated video dimensions ({H_gen}x{W_gen}).")
            # TODO 原video 大小小于生成的video大小，如何resize?

        # Center crop
        # BUG check whether center crop is correct
        start_h = (H_eval - H_gen) // 2
        start_w = (W_eval - W_gen) // 2
        cropped_video = video[:T_gen, :, start_h:start_h + H_gen, start_w:start_w + W_gen]
        
        preprocessed_videos.append(cropped_video)
    
    return preprocessed_videos



def main(args):
    device = torch.device(f"cuda:{args.device}")
    
    eval_video_dir = args.eval_video_dir
    generated_video_dir = args.generated_video_dir

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

    # Check if all generated videos have the same shape
    first_shape = generated_videos[0].shape
    for video in generated_videos:
        if video.shape != first_shape:
            raise ValueError("All generated videos must have the same shape.")


    generated_video_shape = generated_videos[0].shape
    generated_videos = preprocess_eval_videos(eval_videos, generated_video_shape)


    eval_videos_tensor = torch.stack(eval_videos).to(device)
    generated_videos_tensor = torch.stack(generated_videos).to(device)


    print(f"Loaded {len(eval_videos)} evaluation videos | {len(generated_videos)} generated videos")
    
    if args.calculate_lpips:
        result = calculate_lpips(eval_videos_tensor, generated_videos_tensor, device=device)
        
        print("LPIPS results:")
        print(json.dumps(result, indent=4))
        
        output_file = os.path.join(args.generated_video_dir, 'lpips_results.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)

    elif args.calculate_psnr:
        result = calculate_psnr(eval_videos_tensor, generated_videos_tensor)
        
        print("PSNR results:")
        print(json.dumps(result, indent=4))
        
        output_file = os.path.join(args.generated_video_dir, 'psnr_results.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)

    elif args.calculate_ssim:
        result = calculate_ssim(eval_videos_tensor, generated_videos_tensor)
        
        print("SSIM results:")
        print(json.dumps(result, indent=4))
        
        output_file = os.path.join(args.generated_video_dir, 'ssim_results.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)

    elif args.calculate_fvd:
        result = calculate_fvd(eval_videos_tensor, generated_videos_tensor, device=device, method=args.eval_method)
        
        print(f"FVD results with {args.eval_method}:")
        print(json.dumps(result, indent=4))
        
        output_file = os.path.join(args.generated_video_dir, 'fvd_results.json')
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # eval
    parser.add_argument("--calculate_fvd", action="store_true")
    parser.add_argument("--calculate_lpips", action="store_true")
    parser.add_argument("--calculate_psnr", action="store_true")
    parser.add_argument("--calculate_ssim", action="store_true")
    
    parser.add_argument("--eval_method", type=str, default="videogpt")
    
    # dataset
    parser.add_argument("--eval_dataset", type=str, default="./evaluations/fastvideodiffusion/datasets/webvid_selected.csv")
    parser.add_argument("--eval_video_dir", type=str, default="./evaluations/fastvideodiffusion/datasets/webvid")
    parser.add_argument("--generated_video_dir", type=str, default="./evaluations/fastvideodiffusion/samples/latte/sample_skip")
    
    parser.add_argument("--device", type=str, default="0")
    
    args = parser.parse_args()

    main(args)