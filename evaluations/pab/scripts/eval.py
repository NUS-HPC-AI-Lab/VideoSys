import argparse
import importlib
import os
import sys

import imageio
import torch
import torchvision.transforms.functional as F
import tqdm

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "common_metrics_on_video_quality"))

calculate_fvd = importlib.import_module("calculate_fvd").calculate_fvd
calculate_lpips = importlib.import_module("calculate_lpips").calculate_lpips
calculate_psnr = importlib.import_module("calculate_psnr").calculate_psnr
calculate_ssim = importlib.import_module("calculate_ssim").calculate_ssim


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
    reader = imageio.get_reader(video_path, "ffmpeg")

    # Extract frames and convert to a list of tensors
    frames = []
    for frame in reader:
        # Convert the frame to a tensor and permute the dimensions to match (C, H, W)
        frame_tensor = torch.tensor(frame).cuda().permute(2, 0, 1)
        frames.append(frame_tensor)

    # Stack the list of tensors into a single tensor with shape (T, C, H, W)
    video_tensor = torch.stack(frames)

    return video_tensor


def resize_video(video, target_height, target_width):
    resized_frames = []
    for frame in video:
        resized_frame = F.resize(frame, [target_height, target_width])
        resized_frames.append(resized_frame)
    return torch.stack(resized_frames)


def preprocess_eval_video(eval_video, generated_video_shape):
    T_gen, _, H_gen, W_gen = generated_video_shape
    T_eval, _, H_eval, W_eval = eval_video.shape

    if T_eval < T_gen:
        raise ValueError(f"Eval video time steps ({T_eval}) are less than generated video time steps ({T_gen}).")

    if H_eval < H_gen or W_eval < W_gen:
        # Resize the video maintaining the aspect ratio
        resize_height = max(H_gen, int(H_gen * (H_eval / W_eval)))
        resize_width = max(W_gen, int(W_gen * (W_eval / H_eval)))
        eval_video = resize_video(eval_video, resize_height, resize_width)
        # Recalculate the dimensions
        T_eval, _, H_eval, W_eval = eval_video.shape

    # Center crop
    start_h = (H_eval - H_gen) // 2
    start_w = (W_eval - W_gen) // 2
    cropped_video = eval_video[:T_gen, :, start_h : start_h + H_gen, start_w : start_w + W_gen]

    return cropped_video


def main(args):
    device = "cuda"
    eval_video_dir = args.eval_video_dir
    generated_video_dir = args.generated_video_dir

    video_ids = []
    file_extension = "mp4"
    for f in os.listdir(generated_video_dir):
        if f.endswith(f".{file_extension}"):
            video_ids.append(f.split(".")[0])
    if not video_ids:
        raise ValueError("No videos found in the generated video dataset. Exiting.")

    print(f"Find {len(video_ids)} videos")
    prompt_interval = 1
    batch_size = 16

    lpips_results = []
    psnr_results = []
    ssim_results = []
    fvd_results = []

    total_len = len(video_ids) // batch_size + (1 if len(video_ids) % batch_size != 0 else 0)

    for idx, video_id in enumerate(tqdm.tqdm(range(total_len))):
        eval_videos_tensor = []
        generated_videos_tensor = []
        for i in range(batch_size):
            video_idx = idx * batch_size + i
            if video_idx >= len(video_ids):
                break
            video_id = video_ids[video_idx]
            generated_video = load_video(os.path.join(generated_video_dir, f"{video_id}.{file_extension}"))
            generated_videos_tensor.append(generated_video)
            eval_video = load_video(os.path.join(eval_video_dir, f"{video_id}.{file_extension}"))
            eval_videos_tensor.append(eval_video)
        eval_videos_tensor = (torch.stack(eval_videos_tensor) / 255.0).cpu()
        generated_videos_tensor = (torch.stack(generated_videos_tensor) / 255.0).cpu()

        if args.calculate_lpips:
            result = calculate_lpips(eval_videos_tensor, generated_videos_tensor, device=device)
            result = result["value"].values()
            result = sum(result) / len(result)
            lpips_results.append(result)

        if args.calculate_psnr:
            result = calculate_psnr(eval_videos_tensor, generated_videos_tensor)
            result = result["value"].values()
            result = sum(result) / len(result)
            psnr_results.append(result)

        if args.calculate_ssim:
            result = calculate_ssim(eval_videos_tensor, generated_videos_tensor)
            result = result["value"].values()
            result = sum(result) / len(result)
            ssim_results.append(result)

        if args.calculate_fvd:
            result = calculate_fvd(eval_videos_tensor, generated_videos_tensor, device=device, method=args.eval_method)
            result = result["value"].values()
            result = sum(result) / len(result)
            fvd_results.append(result)

        if (idx + 1) % prompt_interval == 0:
            out_str = ""
            for results, name in zip([lpips_results, psnr_results, ssim_results], ["lpips", "psnr", "ssim"]):
                result = sum(results) / len(results)
                out_str += f"{name}: {result:.4f}, "
            print(f"Processed {idx + 1} videos. {out_str[:-2]}")

    out_str = ""
    for results, name in zip([lpips_results, psnr_results, ssim_results], ["lpips", "psnr", "ssim"]):
        result = sum(results) / len(results)
        out_str += f"{name}: {result:.4f}, "
    print(f"Processed all videos. {out_str[:-2]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # eval
    parser.add_argument("--calculate_fvd", action="store_true")
    parser.add_argument("--calculate_lpips", action="store_true")
    parser.add_argument("--calculate_psnr", action="store_true")
    parser.add_argument("--calculate_ssim", action="store_true")
    parser.add_argument("--eval_method", type=str, default="videogpt")

    # dataset
    parser.add_argument("--eval_video_dir", type=str, default="./evaluations/fastvideodiffusion/datasets/webvid")
    parser.add_argument(
        "--generated_video_dir", type=str, default="./evaluations/fastvideodiffusion/samples/latte/sample_skip"
    )

    args = parser.parse_args()

    main(args)
