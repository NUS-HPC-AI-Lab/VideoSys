from einops import rearrange
from torch import nn


class AutoencoderKLWrapper(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.module = vae
        self.out_channels = vae.config.latent_channels
        self.patch_size = [1, 8, 8]

    def encode(self, x):
        # x is (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.module.encode(x).latent_dist.sample().mul_(0.18215)
        x = rearrange(x, "(b t) c h w -> b c t h w", b=B)
        return x

    def decode(self, x):
        # x is (B, C, T, H, W)
        B = x.shape[0]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.module.decode(x / 0.18215).sample
        x = rearrange(x, "(b t) c h w -> b c t h w", b=B)
        return x
