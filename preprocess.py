# This code is modified from https://github.com/wilson1yan/VideoGPT
# Copyright (c) 2021 Wilson Yan. All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import torch
from matplotlib import animation
from matplotlib import pyplot as plt
from torchvision.io import read_video, write_video

from opendit.vqvae.data import preprocess
from opendit.vqvae.download import load_vqvae

video_filename = "/data/personal/nus-zxl/OpenDiT/videos/art-museum.mp4"
sequence_length = 80
resolution = 256
device = torch.device("cuda")

vqvae = load_vqvae("ucf101_stride4x4x4").to(device)
video = read_video(video_filename, pts_unit="sec")[0]
video = preprocess(video, resolution, sequence_length).unsqueeze(0).to(device)

with torch.no_grad():
    encodings = vqvae.encode(video)
    video_recon = vqvae.decode(encodings)
    video_recon = torch.clamp(video_recon, -0.5, 0.5)

videos = video_recon[0].permute(1, 2, 3, 0)
videos = ((videos + 0.5) * 255).cpu().to(torch.uint8)
write_video("output.mp4", videos, 20)


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
