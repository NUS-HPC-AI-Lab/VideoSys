# Modified from Meta DiT

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT:   https://github.com/facebookresearch/DiT/tree/main
# GLIDE: https://github.com/openai/glide-text2im
# MAE:   https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------


from typing import Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint
from timm.models.vision_transformer import Mlp
from torch.distributed import ProcessGroup

from opendit.modules.attn import DistAttention


def get_layernorm(hidden_size: torch.Tensor, eps: float, affine: bool, use_kernel: bool):
    if use_kernel:
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(hidden_size, elementwise_affine=affine, eps=eps)
        except ImportError:
            raise RuntimeError("FusedLayerNorm not available. Please install apex.")
    else:
        return nn.LayerNorm(hidden_size, eps, elementwise_affine=affine)


def modulate(norm_func, x, shift, scale, use_kernel=False):
    # Suppose x is (N, T, D), shift is (N, D), scale is (N, D)
    dtype = x.dtype
    x = norm_func(x.to(torch.float32)).to(dtype)
    if use_kernel:
        try:
            from opendit.kernels.fused_modulate import fused_modulate

            x = fused_modulate(x, scale, shift)
        except ImportError:
            raise RuntimeError("FusedModulate kernel not available. Please install triton.")
    else:
        x = x * (scale.unsqueeze(1) + 1) + shift.unsqueeze(1)
    x = x.to(dtype)

    return x


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        enable_flashattn=False,
        sequence_parallel_size: int = 1,
        sequence_parallel_group: Optional[ProcessGroup] = None,
        sequence_parallel_type: str = None,
        enable_layernorm_kernel=False,
        enable_modulate_kernel=False,
        **block_kwargs,
    ):
        super().__init__()
        self.enable_modulate_kernel = enable_modulate_kernel
        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = DistAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flashattn=enable_flashattn,
            sequence_parallel_size=sequence_parallel_size,
            sequence_parallel_group=sequence_parallel_group,
            sequence_parallel_type=sequence_parallel_type,
            **block_kwargs,
        )
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1, x, shift_msa, scale_msa, self.enable_modulate_kernel)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2, x, shift_mlp, scale_mlp, self.enable_modulate_kernel)
        )
        return x


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final, x, shift, scale)
        x = self.linear(x)
        return x
