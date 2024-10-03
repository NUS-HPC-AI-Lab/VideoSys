import argparse
import os

import imageio
import torch
import torchvision.transforms.functional as F
import tqdm
from calculate_lpips import calculate_lpips
from calculate_psnr import calculate_psnr
from calculate_ssim import calculate_ssim


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


def resize_gt_video(gt_video, gen_video):
    gen_video_shape = gen_video.shape
    T_gen, _, H_gen, W_gen = gen_video_shape
    T_eval, _, H_eval, W_eval = gt_video.shape

    if T_eval < T_gen:
        raise ValueError(f"Eval video time steps ({T_eval}) are less than generated video time steps ({T_gen}).")

    if H_eval < H_gen or W_eval < W_gen:
        # Resize the video maintaining the aspect ratio
        resize_height = max(H_gen, int(H_gen * (H_eval / W_eval)))
        resize_width = max(W_gen, int(W_gen * (W_eval / H_eval)))
        gt_video = resize_video(gt_video, resize_height, resize_width)
        # Recalculate the dimensions
        T_eval, _, H_eval, W_eval = gt_video.shape

    # Center crop
    start_h = (H_eval - H_gen) // 2
    start_w = (W_eval - W_gen) // 2
    cropped_video = gt_video[:T_gen, :, start_h : start_h + H_gen, start_w : start_w + W_gen]

    return cropped_video


def get_video_ids(gt_video_dirs, gen_video_dirs):
    video_ids = []
    for f in os.listdir(gt_video_dirs[0]):
        if f.endswith(f".mp4"):
            video_ids.append(f.replace(f".mp4", ""))
    video_ids.sort()

    for video_dir in gt_video_dirs + gen_video_dirs:
        tmp_video_ids = []
        for f in os.listdir(video_dir):
            if f.endswith(f".mp4"):
                tmp_video_ids.append(f.replace(f".mp4", ""))
        tmp_video_ids.sort()
        if tmp_video_ids != video_ids:
            raise ValueError(f"Video IDs in {video_dir} are different.")
    return video_ids


def get_videos(video_ids, gt_video_dirs, gen_video_dirs):
    gt_videos = {}
    generated_videos = {}

    for gt_video_dir in gt_video_dirs:
        tmp_gt_videos_tensor = []
        for video_id in video_ids:
            gt_video = load_video(os.path.join(gt_video_dir, f"{video_id}.mp4"))
            tmp_gt_videos_tensor.append(gt_video)
        gt_videos[gt_video_dir] = tmp_gt_videos_tensor

    for generated_video_dir in gen_video_dirs:
        tmp_generated_videos_tensor = []
        for video_id in video_ids:
            generated_video = load_video(os.path.join(generated_video_dir, f"{video_id}.mp4"))
            tmp_generated_videos_tensor.append(generated_video)
        generated_videos[generated_video_dir] = tmp_generated_videos_tensor

    return gt_videos, generated_videos


def print_results(lpips_results, psnr_results, ssim_results, gt_video_dirs, gen_video_dirs):
    out_str = ""

    for gt_video_dir in gt_video_dirs:
        for generated_video_dir in gen_video_dirs:
            if gt_video_dir == generated_video_dir:
                continue
            lpips = sum(lpips_results[gt_video_dir][generated_video_dir]) / len(
                lpips_results[gt_video_dir][generated_video_dir]
            )
            psnr = sum(psnr_results[gt_video_dir][generated_video_dir]) / len(
                psnr_results[gt_video_dir][generated_video_dir]
            )
            ssim = sum(ssim_results[gt_video_dir][generated_video_dir]) / len(
                ssim_results[gt_video_dir][generated_video_dir]
            )
            out_str += f"\ngt: {gt_video_dir} -> gen: {generated_video_dir}, lpips: {lpips:.4f}, psnr: {psnr:.4f}, ssim: {ssim:.4f}"

    return out_str


def main(args):
    device = "cuda"
    gt_video_dirs = args.gt_video_dirs
    gen_video_dirs = args.gen_video_dirs

    video_ids = get_video_ids(gt_video_dirs, gen_video_dirs)
    print(f"Find {len(video_ids)} videos")

    prompt_interval = 1
    batch_size = 8
    calculate_lpips_flag, calculate_psnr_flag, calculate_ssim_flag = True, True, True

    lpips_results = {}
    psnr_results = {}
    ssim_results = {}
    for gt_video_dir in gt_video_dirs:
        lpips_results[gt_video_dir] = {}
        psnr_results[gt_video_dir] = {}
        ssim_results[gt_video_dir] = {}
        for generated_video_dir in gen_video_dirs:
            lpips_results[gt_video_dir][generated_video_dir] = []
            psnr_results[gt_video_dir][generated_video_dir] = []
            ssim_results[gt_video_dir][generated_video_dir] = []

    total_len = len(video_ids) // batch_size + (1 if len(video_ids) % batch_size != 0 else 0)

    for idx in tqdm.tqdm(range(total_len)):
        video_ids_batch = video_ids[idx * batch_size : (idx + 1) * batch_size]
        gt_videos, generated_videos = get_videos(video_ids_batch, gt_video_dirs, gen_video_dirs)

        for gt_video_dir, gt_videos_tensor in gt_videos.items():
            for generated_video_dir, generated_videos_tensor in generated_videos.items():
                if gt_video_dir == generated_video_dir:
                    continue

                if not isinstance(gt_videos_tensor, torch.Tensor):
                    for i in range(len(gt_videos_tensor)):
                        gt_videos_tensor[i] = resize_gt_video(gt_videos_tensor[i], generated_videos_tensor[0])
                    gt_videos_tensor = (torch.stack(gt_videos_tensor) / 255.0).cpu()

                generated_videos_tensor = (torch.stack(generated_videos_tensor) / 255.0).cpu()

                if calculate_lpips_flag:
                    result = calculate_lpips(gt_videos_tensor, generated_videos_tensor, device=device)
                    result = result["value"].values()
                    result = float(sum(result) / len(result))
                    lpips_results[gt_video_dir][generated_video_dir].append(result)

                if calculate_psnr_flag:
                    result = calculate_psnr(gt_videos_tensor, generated_videos_tensor)
                    result = result["value"].values()
                    result = float(sum(result) / len(result))
                    psnr_results[gt_video_dir][generated_video_dir].append(result)

                if calculate_ssim_flag:
                    result = calculate_ssim(gt_videos_tensor, generated_videos_tensor)
                    result = result["value"].values()
                    result = float(sum(result) / len(result))
                    ssim_results[gt_video_dir][generated_video_dir].append(result)

        if (idx + 1) % prompt_interval == 0:
            out_str = print_results(lpips_results, psnr_results, ssim_results, gt_video_dirs, gen_video_dirs)
            print(f"Processed {idx + 1} / {total_len} videos. {out_str}")

    out_str = print_results(lpips_results, psnr_results, ssim_results, gt_video_dirs, gen_video_dirs)

    # save
    with open(f"./batch_eval.txt", "w+") as f:
        f.write(out_str)

    print(f"Processed all videos. {out_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_video_dirs", type=str, nargs="+")
    parser.add_argument("--gen_video_dirs", type=str, nargs="+")

    args = parser.parse_args()

    main(args)
