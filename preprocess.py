# This code is modified from https://github.com/wilson1yan/VideoGPT
# Copyright (c) 2021 Wilson Yan. All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import torch
import tqdm
from matplotlib import animation
from matplotlib import pyplot as plt
from torch import nn
from torchvision.io import read_video, write_video

from opendit.vqvae.data import preprocess
from opendit.vqvae.download import load_vqvae

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@torch.no_grad()
def visualize(path: str) -> None:
    video_filename = path
    sequence_length = 80
    resolution = 256
    device = torch.device("cuda")

    # build model and load weights
    vqvae = load_vqvae("ucf101_stride4x4x4").to(device)

    # read video and preprocess
    video = read_video(video_filename, pts_unit="sec")[0]
    video = preprocess(video, resolution, sequence_length).unsqueeze(0).to(device)

    # encode and decode
    encodings = vqvae.encode(video)
    video_recon = vqvae.decode(encodings)
    video_recon = torch.clamp(video_recon, -0.5, 0.5)

    # save reconstruction video
    videos = video_recon[0].permute(1, 2, 3, 0)
    videos = ((videos + 0.5) * 255).cpu().to(torch.uint8)
    write_video("output.mp4", videos, 20)

    # compare real and reconstruction video
    videos = torch.cat((video, video_recon), dim=-1)
    videos = videos[0].permute(1, 2, 3, 0)  # CTHW -> THWC
    videos = ((videos + 0.5) * 255).cpu().to(torch.uint8).numpy()
    fig = plt.figure()
    plt.title("real (left), reconstruction (right)")
    plt.axis("off")
    im = plt.imshow(videos[0, :, :, :])
    plt.close()

    def init():
        im.set_data(videos[0, :, :, :])

    def animate(i):
        im.set_data(videos[i, :, :, :])
        return im

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=videos.shape[0], interval=50)
    mp4_writer = animation.FFMpegWriter(fps=20, bitrate=1800)
    anim.save("compare.mp4", writer=mp4_writer)


@torch.no_grad()
def encode_video(
    model: nn.Module,
    file_path: str,
    save_dir: str,
    sequence_length: int = 32,
    resolution: int = 256,
    video_type: str = ".mp4",
) -> None:
    video_filename = file_path
    assert video_filename.endswith(video_type)
    device = torch.device("cuda")

    # read video and preprocess
    video = read_video(video_filename, pts_unit="sec")[0]
    video = preprocess(video, resolution, sequence_length).unsqueeze(0).to(device)

    # encode
    encodings = model.encode(video, include_embeddings=True)[1]
    encodings = encodings.cpu().numpy()

    # save encodings
    os.makedirs(save_dir, exist_ok=True)
    embed_path = os.path.join(save_dir, os.path.basename(file_path).replace(video_type, ".npy"))
    with open(embed_path, "wb") as f:
        f.write(encodings.tobytes())


def preprocess_video(args):
    vqvae = load_vqvae(args.model).to("cuda")
    video_list = os.listdir(args.data_dir)
    for v in tqdm.tqdm(video_list):
        if v.endswith(args.video_type):
            encode_video(
                model=vqvae, file_path=os.path.join(args.data_dir, v), save_dir="processed", video_type=args.video_type
            )

    print("Done!")


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="ucf101_stride4x4x4")
    parser.add_argument("--video_type", type=str, default=".mp4")
    args = parser.parse_args()
    print(f"Args: {args}")
    preprocess_video(args)
    # visualize("./videos/art-museum.mp4")
