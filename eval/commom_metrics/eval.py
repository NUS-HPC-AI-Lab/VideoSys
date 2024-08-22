import argparse
import os

import imageio
import torch
import torchvision.transforms.functional as F
import tqdm
from calculate_lpips import calculate_lpips
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim


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
    gt_video_dir = args.gt_video_dir
    generated_video_dir = args.generated_video_dir

    video_ids = []
    file_extension = "mp4"
    for f in os.listdir(generated_video_dir):
        if f.endswith(f".{file_extension}"):
            video_ids.append(f.replace(f".{file_extension}", ""))
    if not video_ids:
        raise ValueError("No videos found in the generated video dataset. Exiting.")

    print(f"Find {len(video_ids)} videos")
    prompt_interval = 1
    batch_size = 16
    calculate_lpips_flag, calculate_psnr_flag, calculate_ssim_flag = True, True, True

    lpips_results = []
    psnr_results = []
    ssim_results = []

    total_len = len(video_ids) // batch_size + (1 if len(video_ids) % batch_size != 0 else 0)

    for idx, video_id in enumerate(tqdm.tqdm(range(total_len))):
        gt_videos_tensor = []
        generated_videos_tensor = []
        for i in range(batch_size):
            video_idx = idx * batch_size + i
            if video_idx >= len(video_ids):
                break
            video_id = video_ids[video_idx]
            generated_video = load_video(os.path.join(generated_video_dir, f"{video_id}.{file_extension}"))
            generated_videos_tensor.append(generated_video)
            eval_video = load_video(os.path.join(gt_video_dir, f"{video_id}.{file_extension}"))
            gt_videos_tensor.append(eval_video)
        gt_videos_tensor = (torch.stack(gt_videos_tensor) / 255.0).cpu()
        generated_videos_tensor = (torch.stack(generated_videos_tensor) / 255.0).cpu()

        if calculate_lpips_flag:
            result = calculate_lpips(gt_videos_tensor, generated_videos_tensor, device=device)
            result = result["value"].values()
            result = sum(result) / len(result)
            lpips_results.append(result)

        if calculate_psnr_flag:
            result = calculate_psnr(gt_videos_tensor, generated_videos_tensor)
            result = result["value"].values()
            result = sum(result) / len(result)
            psnr_results.append(result)

        if calculate_ssim_flag:
            result = calculate_ssim(gt_videos_tensor, generated_videos_tensor)
            result = result["value"].values()
            result = sum(result) / len(result)
            ssim_results.append(result)

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
    out_str = out_str[:-2]

    # save
    with open(f"./{os.path.basename(generated_video_dir)}.txt", "w+") as f:
        f.write(out_str)

    print(f"Processed all videos. {out_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_video_dir", type=str)
    parser.add_argument("--generated_video_dir", type=str)

    args = parser.parse_args()

    main(args)
