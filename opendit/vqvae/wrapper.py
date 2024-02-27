import torch
from torch import nn


class AutoencoderKLWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.module = vae
        self.out_channels = vae.config.latent_channels
        self.patch_size = [1, 8, 8]

    def encode(self, x):
        # x is (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x = torch.einsum("bcthw->btcwh", x)
        x = x.contiguous()
        x = x.view(B * T, C, H, W)
        x = self.module.encode(x).latent_dist.sample().mul_(0.18215)
        x = x.view(B, T, *x.shape[1:])
        x = torch.einsum("btcwh->bcthw", x)
        return x

    def decode(self, x):
        # x is (B, C, T, H, W)
        B, C, T, H, W = x.shape
        x = torch.einsum("bcthw->btcwh", x)
        x = x.contiguous()
        x = x.view(B * T, C, H, W)
        x = self.module.decode(x / 0.18215).sample
        x = x.view(B, T, *x.shape[1:])
        x = torch.einsum("btcwh->bcthw", x)
        return x
