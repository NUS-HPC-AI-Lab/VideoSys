# Modified from Meta DiT

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT:   https://github.com/facebookresearch/DiT/tree/main
# GLIDE: https://github.com/openai/glide-text2im
# MAE:   https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import math
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from timm.models.vision_transformer import Mlp, PatchEmbed
from torch.distributed import ProcessGroup
from torch.jit import Final

from opendit.models.clip import TextEmbedder
from opendit.utils.operation import AllGather, AsyncAllGatherForTwo, all_to_all_comm, gather_forward_split_backward


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


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(
            device=t.device
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if t_freq.dtype != dtype:
            t_freq = t_freq.to(dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core DiT Model                                #
#################################################################################


class DistAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        enable_flashattn: bool = False,
        sequence_parallel_size: int = 1,
        sequence_parallel_group: Optional[ProcessGroup] = None,
        sequence_parallel_type: str = None,
        sequence_parallel_overlap: bool = False,
        sequence_parallel_overlap_size: int = 2,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.enable_flashattn = enable_flashattn

        # sequence parallel
        self.sequence_parallel_type = sequence_parallel_type
        self.sequence_parallel_size = sequence_parallel_size
        if sequence_parallel_size == 1:
            sequence_parallel_type = None
        else:
            assert sequence_parallel_type in [
                "longseq",
                "ulysses",
            ], "sequence_parallel_type should be longseq or ulysses"
        self.sequence_parallel_group = sequence_parallel_group
        self.sequence_parallel_overlap = sequence_parallel_overlap
        self.sequence_parallel_overlap_size = sequence_parallel_overlap_size
        self.sequence_parallel_rank = dist.get_rank(sequence_parallel_group)
        self.sequence_parallel_param_slice = slice(
            self.qkv.out_features // sequence_parallel_size * self.sequence_parallel_rank,
            self.qkv.out_features // sequence_parallel_size * (self.sequence_parallel_rank + 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        total_N = N * self.sequence_parallel_size

        if self.sequence_parallel_type == "longseq":
            if self.sequence_parallel_overlap:
                if self.sequence_parallel_size == 2:
                    # (B, N / SP_SIZE, C) => (SP_SIZE * B, N / SP_SIZE, C)
                    qkv = AsyncAllGatherForTwo.apply(
                        x,
                        self.qkv.weight[self.sequence_parallel_param_slice],
                        self.qkv.bias[self.sequence_parallel_param_slice],
                        self.sequence_parallel_rank,
                        self.sequence_parallel_size,
                        dist.group.WORLD,
                    )  # (B, N, C / SP_SIZE)
                else:
                    raise NotImplementedError(
                        "sequence_parallel_overlap is only supported for sequence_parallel_size=2"
                    )
            else:
                # (B, N / SP_SIZE, C) => (SP_SIZE * B, N / SP_SIZE, C)
                x = AllGather.apply(x)[0]
                # (SP_SIZE, B, N / SP_SIZE, C) => (B, N, C)
                x = rearrange(x, "sp b n c -> b (sp n) c")
                qkv = F.linear(
                    x,
                    self.qkv.weight[self.sequence_parallel_param_slice],
                    self.qkv.bias[self.sequence_parallel_param_slice],
                )
        else:
            qkv = self.qkv(x)  # (B, N, C), N here is N_total // SP_SIZE

        num_heads = (
            self.num_heads if self.sequence_parallel_type is None else self.num_heads // self.sequence_parallel_size
        )

        if self.sequence_parallel_type == "ulysses":
            q, k, v = qkv.split(self.head_dim * self.num_heads, dim=-1)
            q = all_to_all_comm(q, self.sequence_parallel_group)
            k = all_to_all_comm(k, self.sequence_parallel_group)
            v = all_to_all_comm(v, self.sequence_parallel_group)

            if self.enable_flashattn:
                q = q.reshape(B, N * self.sequence_parallel_size, num_heads, self.head_dim).contiguous()
                k = k.reshape(B, N * self.sequence_parallel_size, num_heads, self.head_dim).contiguous()
                v = v.reshape(B, N * self.sequence_parallel_size, num_heads, self.head_dim).contiguous()
            else:
                q = (
                    q.reshape(B, N * self.sequence_parallel_size, num_heads, self.head_dim)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )
                k = (
                    k.reshape(B, N * self.sequence_parallel_size, num_heads, self.head_dim)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )
                v = (
                    v.reshape(B, N * self.sequence_parallel_size, num_heads, self.head_dim)
                    .permute(0, 2, 1, 3)
                    .contiguous()
                )

        else:
            if self.sequence_parallel_type == "longseq":
                qkv_shape = (B, total_N, num_heads, 3, self.head_dim)
                if self.enable_flashattn:
                    qkv_permute_shape = (3, 0, 1, 2, 4)
                else:
                    qkv_permute_shape = (3, 0, 2, 1, 4)
            else:
                qkv_shape = (B, total_N, 3, num_heads, self.head_dim)
                if self.enable_flashattn:
                    qkv_permute_shape = (2, 0, 1, 3, 4)
                else:
                    qkv_permute_shape = (2, 0, 3, 1, 4)
            qkv = qkv.view(qkv_shape).permute(qkv_permute_shape)
            q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        if self.enable_flashattn:
            from flash_attn import flash_attn_func

            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
            )
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            # translate attn to float32
            attn = attn.to(torch.float32)
            attn = attn.softmax(dim=-1)
            # cast back attn to original dtype
            attn = attn.to(dtype)
            attn = self.attn_drop(attn)
            x = attn @ v

        if self.sequence_parallel_type is None:
            x_output_shape = (B, N, C)
        else:
            x_output_shape = (B, total_N, num_heads * self.head_dim)
        if self.enable_flashattn:
            x = x.reshape(x_output_shape)
        else:
            x = x.transpose(1, 2).reshape(x_output_shape)
        if self.sequence_parallel_size > 1:
            x = all_to_all_comm(x, self.sequence_parallel_group, scatter_dim=1, gather_dim=2)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def rearrange_fused_weight(self, layer: nn.Linear, flag="load"):
        # check whether layer is an torch.nn.Linear layer
        if not isinstance(layer, nn.Linear):
            raise ValueError("Invalid layer type for fused qkv weight rearrange!")

        with torch.no_grad():
            if flag == "load":
                layer.weight.data = rearrange(layer.weight.data, "(x NH H) D -> (NH x H) D", x=3, H=self.head_dim)
                layer.bias.data = rearrange(layer.bias.data, "(x NH H) -> (NH x H)", x=3, H=self.head_dim)
                assert layer.weight.data.is_contiguous()
                assert layer.bias.data.is_contiguous()

            elif flag == "save":
                layer.weight.data = rearrange(layer.weight.data, "(NH x H) D -> (x NH H) D", x=3, H=self.head_dim)
                layer.bias.data = rearrange(layer.bias.data, "(NH x H) -> (x NH H)", x=3, H=self.head_dim)
                assert layer.weight.data.is_contiguous()
                assert layer.bias.data.is_contiguous()
            else:
                raise ValueError("Invalid flag for fused qkv weight rearrange!")


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
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final, x, shift, scale)
        x = self.linear(x)
        return x


class DiT(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """

    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        text_condition=None,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma: bool = True,
        enable_flashattn: bool = False,
        enable_layernorm_kernel: bool = False,
        enable_modulate_kernel: bool = False,
        sequence_parallel_size: int = 1,
        sequence_parallel_group: Optional[ProcessGroup] = None,
        sequence_parallel_type: str = None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.sequence_parallel_size = sequence_parallel_size
        self.sequence_parallel_group = sequence_parallel_group
        self.sequence_parallel_type = sequence_parallel_type

        self.dtype = dtype
        if enable_flashattn:
            assert dtype in [
                torch.float16,
                torch.bfloat16,
            ], f"Flash attention only supports float16 and bfloat16, but got {self.dtype}"

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        self.text_condition = text_condition
        if text_condition is not None:
            self.y_embedder = TextEmbedder(path=text_condition, hidden_size=hidden_size)
        else:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    hidden_size,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    enable_flashattn=enable_flashattn,
                    enable_modulate_kernel=enable_modulate_kernel,
                    enable_layernorm_kernel=enable_layernorm_kernel,
                    sequence_parallel_size=self.sequence_parallel_size,
                    sequence_parallel_group=self.sequence_parallel_group,
                    sequence_parallel_type=self.sequence_parallel_type,
                )
                for _ in range(depth)
            ]
        )
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

        self.gradient_checkpointing = False
        self.use_flash_attention = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize label embedding table:
        if self.text_condition is None:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    @staticmethod
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)

        return custom_forward

    def forward(self, x, t, y):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """

        # origin inputs should be float32, cast to specified dtype
        x = x.to(self.dtype)

        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t, dtype=x.dtype)  # (N, D)
        y = self.y_embedder(y, self.training)  # (N, D)
        c = t + y  # (N, D)

        # Chunk x on sequence dimension to sp group
        if self.sequence_parallel_size > 1:
            x = x.chunk(self.sequence_parallel_size, dim=1)[dist.get_rank(self.sequence_parallel_group)]

        for block in self.blocks:
            if self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(self.create_custom_forward(block), x, c)
            else:
                x = block(x, c)  # (N, T, D)

        if self.sequence_parallel_size > 1:
            x = gather_forward_split_backward(x, dim=1, process_group=self.sequence_parallel_group)

        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale):
        """
        Forward pass of DiT, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, y)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

    def rearrange_attention_weights(self, flag="load"):
        for block in self.blocks:
            block.attn.rearrange_fused_weight(block.attn.qkv, flag)
        torch.cuda.empty_cache()


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def DiT_XL_2(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)


def DiT_XL_4(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)


def DiT_XL_8(**kwargs):
    return DiT(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)


def DiT_L_2(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)


def DiT_L_4(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)


def DiT_L_8(**kwargs):
    return DiT(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)


def DiT_B_2(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)


def DiT_B_4(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)


def DiT_B_8(**kwargs):
    return DiT(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)


def DiT_S_2(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)


def DiT_S_4(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)


def DiT_S_8(**kwargs):
    return DiT(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)


DiT_models = {
    "DiT-XL/2": DiT_XL_2,
    "DiT-XL/4": DiT_XL_4,
    "DiT-XL/8": DiT_XL_8,
    "DiT-L/2": DiT_L_2,
    "DiT-L/4": DiT_L_4,
    "DiT-L/8": DiT_L_8,
    "DiT-B/2": DiT_B_2,
    "DiT-B/4": DiT_B_4,
    "DiT-B/8": DiT_B_8,
    "DiT-S/2": DiT_S_2,
    "DiT-S/4": DiT_S_4,
    "DiT-S/8": DiT_S_8,
}
