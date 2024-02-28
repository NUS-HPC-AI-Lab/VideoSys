import os

import torch
from diffusers.models import AutoencoderKL
from torchvision.io import write_video
from torchvision.utils import save_image

from opendit.vae.wrapper import AutoencoderKLWrapper


def t2v(x):
    x = (x * 0.5 + 0.5).clamp(0, 1)
    x = (x * 255).to(torch.uint8)
    x = x.permute(1, 2, 3, 0).cpu()
    return x


def save_sample(x, real=None):
    B = x.size(0)
    nrows = B // int(B**0.5)
    if x.size(2) == 1:
        path = "sample.png"
        x = x.squeeze(2)
        if real is not None:
            real = real.squeeze(2)
            x = torch.cat([real, x], dim=-1)
        save_image(x, path, nrow=nrows, normalize=True, value_range=(-1, 1))
        print(f"Sampled images saved to {path}")
    else:
        path_dir = "sample_videos"
        os.makedirs(path_dir, exist_ok=True)
        for i in range(B):
            path = os.path.join(path_dir, f"sample_{i}.mp4")
            x_i = t2v(x[i])
            if real is not None:
                real_i = t2v(real[i])
                x_i = torch.cat([real_i, x_i], dim=-2)
            write_video(path, x_i, fps=20, video_codec="h264")
            print(f"Sampled video saved to {path}")


@torch.no_grad()
def reconstruct(args, data) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vae = AutoencoderKL.from_pretrained(args.vae)
    vae = AutoencoderKLWrapper(vae)
    vae = vae.to(device)
    data = data.to(device)

    x = vae.encode(data)
    x = vae.decode(x)
    save_sample(x, real=data)
