import argparse
import os

import imageio
import torch
import torchvision.transforms.functional as F


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
        frame_tensor = torch.tensor(frame).permute(2, 0, 1)
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


def preprocess_eval_videos(eval_videos, generated_video_shape):
    T_gen, C_gen, H_gen, W_gen = generated_video_shape
    preprocessed_videos = []

    for video in eval_videos:
        T_eval, C_eval, H_eval, W_eval = video.shape

        if T_eval < T_gen:
            raise ValueError(f"Eval video time steps ({T_eval}) are less than generated video time steps ({T_gen}).")

        if H_eval < H_gen or W_eval < W_gen:
            # raise ValueError(f"Eval video dimensions ({H_eval}x{W_eval}) are less than generated video dimensions ({H_gen}x{W_gen}).")
            print(
                f"Eval video dimensions ({H_eval}x{W_eval}) are less than generated video dimensions ({H_gen}x{W_gen})."
            )
            # TODO 原video 大小小于生成的video大小，如何resize?
            # Resize the video maintaining the aspect ratio
            resize_height = max(H_gen, int(H_gen * (H_eval / W_eval)))
            resize_width = max(W_gen, int(W_gen * (W_eval / H_eval)))
            resized_video = resize_video(video, resize_height, resize_width)

            # Recalculate the dimensions
            T_eval, C_eval, H_eval, W_eval = resized_video.shape

            # Center crop
            # BUG check whether center crop is correct
            start_h = (H_eval - H_gen) // 2
            start_w = (W_eval - W_gen) // 2
            cropped_video = resized_video[:T_gen, :, start_h : start_h + H_gen, start_w : start_w + W_gen]

            preprocessed_videos.append(cropped_video)

            return preprocessed_videos[0]


def main(args):
    device = torch.device(f"cuda:{args.device}")

    eval_video_dir = args.eval_video_dir
    generated_video_dir = args.generated_video_dir

    video_ids = []
    for f in os.listdir(generated_video_dir):
        if f.endswith(".video"):
            video_ids.append(f.split(".")[0])
            file_extension = "video"
        elif f.endswith(".mp4"):
            video_ids.append(f.split(".")[0])
            file_extension = "mp4"
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
    resized_video = preprocess_eval_videos(eval_videos, generated_video_shape)

    imageio.mimwrite("evaluations/fastvideodiffusion/samples/resize.mp4", resized_video.permute(0, 2, 3, 1), fps=8)

    # imageio.mimwrite(
    # "evaluations/fastvideodiffusion/samples/origin.mp4", origin_video, fps=8
    # )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # eval
    parser.add_argument("--calculate_fvd", action="store_true")
    parser.add_argument("--calculate_lpips", action="store_true")
    parser.add_argument("--calculate_psnr", action="store_true")
    parser.add_argument("--calculate_ssim", action="store_true")

    parser.add_argument("--eval_method", type=str, default="videogpt")

    # dataset
    parser.add_argument(
        "--eval_dataset", type=str, default="./evaluations/fastvideodiffusion/datasets/webvid_selected.csv"
    )
    parser.add_argument("--eval_video_dir", type=str, default="./evaluations/fastvideodiffusion/datasets/webvid")
    parser.add_argument(
        "--generated_video_dir", type=str, default="./evaluations/fastvideodiffusion/samples/latte/sample_skip"
    )

    parser.add_argument("--device", type=str, default="0")

    args = parser.parse_args()

    main(args)
