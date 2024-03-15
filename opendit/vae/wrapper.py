# Adapted from OpenSora

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# OpenSora: https://github.com/hpcaitech/Open-Sora
# --------------------------------------------------------


import torch
from diffusers.models import AutoencoderKL
from einops import rearrange
from torch import nn


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
        for i in range(3):
            assert input_size[i] % self.patch_size[i] == 0, "Input size must be divisible by patch size"
        input_size = [input_size[i] // self.patch_size[i] for i in range(3)]
        return input_size
