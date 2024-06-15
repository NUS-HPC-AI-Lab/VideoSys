# Adapted from DiT and OpenSora

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT:      https://github.com/facebookresearch/DiT
# OpenSora: https://github.com/hpcaitech/Open-Sora
# --------------------------------------------------------


import torch
from diffusers.models import AutoencoderKL
from einops import rearrange
from torch import nn
from torchvision.io import write_video
from torchvision.utils import save_image


class VideoAutoencoderKL(nn.Module):
    def __init__(self, from_pretrained=None, split=None):
        super().__init__()
        self.module = AutoencoderKL.from_pretrained(from_pretrained)
        self.out_channels = self.module.config.latent_channels
        self.patch_size = (1, 8, 8)
        self.split = split

    def encode(self, x):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")

        if self.split is None:
            x = self.module.encode(x).latent_dist.sample().mul_(0.18215)
        else:
            bs = x.shape[0] // self.split
            x_out = []
            for i in range(self.split):
                x_out.append(self.module.encode(x[i * bs : (i + 1) * bs]).latent_dist.sample().mul_(0.18215))
            x = torch.cat(x_out, dim=0)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def decode(self, x):
        # x: (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "B C T H W -> (B T) C H W")
        if self.split is None:
            x = self.module.decode(x / 0.18215).sample
        else:
            bs = x.shape[0] // self.split
            x_out = []
            for i in range(self.split):
                x_out.append(self.module.decode(x[i * bs : (i + 1) * bs] / 0.18215).sample)
            x = torch.cat(x_out, dim=0)
        x = rearrange(x, "(B T) C H W -> B C T H W", B=B)
        return x

    def get_latent_size(self, input_size):
        latent_size = []
        for i in range(3):
            # assert (
            #     input_size[i] is None or input_size[i] % self.patch_size[i] == 0
            # ), "Input size must be divisible by patch size"
            latent_size.append(input_size[i] // self.patch_size[i] if input_size[i] is not None else None)
        return latent_size


def save_sample(x, fps=8, save_path=None, normalize=True, value_range=(-1, 1)):
    """
    Args:
        x (Tensor): shape [C, T, H, W]
    """
    assert x.ndim == 4

    if x.shape[1] == 1:  # T = 1: save as image
        save_path += ".png"
        x = x.squeeze(1)
        save_image([x], save_path, normalize=normalize, value_range=value_range)
    else:
        save_path += ".mp4"
        if normalize:
            low, high = value_range
            x.clamp_(min=low, max=high)
            x.sub_(low).div_(max(high - low, 1e-5))

        x = x.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 3, 0).to("cpu", torch.uint8)
        write_video(save_path, x, fps=fps, video_codec="h264")
    print(f"Saved to {save_path}")
