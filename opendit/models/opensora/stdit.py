# Adapted from OpenSora and DiT

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DiT:      https://github.com/facebookresearch/DiT
# OpenSora: https://github.com/hpcaitech/Open-Sora
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp

from opendit.core.comm import all_to_all_comm, gather_sequence, split_sequence
from opendit.core.parallel_mgr import get_sequence_parallel_group, get_sequence_parallel_size, use_sequecne_parallelism
from opendit.embed.clip_text_emb import CaptionEmbedder
from opendit.embed.patch_emb import PatchEmbed3D
from opendit.embed.pos_emb import get_1d_sincos_pos_embed, get_2d_sincos_pos_embed
from opendit.embed.time_emb import TimestepEmbedder
from opendit.models.opensora.ckpt_io import load_checkpoint
from opendit.modules.attn import Attention, MultiHeadCrossAttention
from opendit.modules.layers import get_layernorm


def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift


def approx_gelu():
    return nn.GELU(approximate="tanh")


class STDiTFinalLayer(nn.Module):
    """
    The final layer of PixArt.
    """

    def __init__(self, hidden_size, num_patch, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, num_patch * out_channels, bias=True)
        self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size) / hidden_size**0.5)
        self.out_channels = out_channels

    def forward(self, x, t):
        shift, scale = (self.scale_shift_table[None] + t[:, None]).chunk(2, dim=1)
        x = t2i_modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class STDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        d_s=None,
        d_t=None,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.enable_flashattn = enable_flashattn
        self.enable_sequence_parallelism = use_sequecne_parallelism()

        self.attn_cls = Attention
        self.mha_cls = MultiHeadCrossAttention

        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flashattn=enable_flashattn,
        )
        self.cross_attn = self.mha_cls(hidden_size, num_heads, enable_flashattn=enable_flashattn)
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = Mlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

        # spatial temporal size
        self.d_s = d_s
        self.d_t = d_t

        self.attn_temp = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flashattn=self.enable_flashattn,
        )

    def forward(self, x, y, t, mask=None, tpe=None):
        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)

        # spatial branch
        d_s, d_t = self.get_spatial_temporal_size(self.enable_sequence_parallelism, True)
        x_s = rearrange(x_m, "b (t s) d -> (b t) s d", t=d_t, s=d_s)
        x_s = self.attn(x_s)
        x_s = rearrange(x_s, "(b t) s d -> b (t s) d", t=d_t, s=d_s)
        x = x + self.drop_path(gate_msa * x_s)

        # temporal to spatial switch
        if self.enable_sequence_parallelism:
            # b t/n s d -> b t s/n d
            x, d_s, d_t = self.dynamic_switch(x, d_s, d_t, temporal_to_spatial=True)

        # temporal branch
        x_t = rearrange(x, "b (t s) d -> (b s) t d", t=d_t, s=d_s)
        if tpe is not None:
            x_t = x_t + tpe
        x_t = self.attn_temp(x_t)
        x_t = rearrange(x_t, "(b s) t d -> b (t s) d", t=d_t, s=d_s)
        x = x + self.drop_path(gate_msa * x_t)

        # spatial to temporal switch
        if self.enable_sequence_parallelism:
            # b t s/n d -> b t/n s d
            x, d_s, d_t = self.dynamic_switch(x, d_s, d_t, temporal_to_spatial=False)

        # cross attn
        x = x + self.cross_attn(x, y, mask)

        # mlp
        x = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return x

    def get_spatial_temporal_size(self, enable_sequence_parallelism: bool, split_temporal: bool):
        if enable_sequence_parallelism:
            if split_temporal:
                return self.d_s, self.d_t // get_sequence_parallel_size()
            else:
                return self.d_s // get_sequence_parallel_size(), self.d_t
        else:
            return self.d_s, self.d_t

    def dynamic_switch(self, x, d_s, d_t, temporal_to_spatial: bool):
        if temporal_to_spatial:
            scatter_dim, gather_dim = 2, 1
            split_temporal = False
        else:
            scatter_dim, gather_dim = 1, 2
            split_temporal = True

        x = rearrange(x, "b (t s) d -> b t s d", t=d_t, s=d_s)
        x = all_to_all_comm(x, get_sequence_parallel_group(), scatter_dim=scatter_dim, gather_dim=gather_dim)
        d_s, d_t = self.get_spatial_temporal_size(self.enable_sequence_parallelism, split_temporal=split_temporal)
        x = rearrange(x, "b t s d -> b (t s) d", t=d_t, s=d_s)
        return x, d_s, d_t


class STDiT(nn.Module):
    def __init__(
        self,
        input_size=(1, 32, 32),
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path=0.0,
        no_temporal_pos_emb=False,
        caption_channels=4096,
        model_max_length=120,
        dtype=torch.float32,
        space_scale=1.0,
        time_scale=1.0,
        freeze=None,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
    ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.input_size = input_size
        num_patches = np.prod([input_size[i] // patch_size[i] for i in range(3)])
        self.num_patches = num_patches
        self.num_temporal = input_size[0] // patch_size[0]
        self.num_spatial = num_patches // self.num_temporal
        self.num_heads = num_heads
        self.dtype = dtype
        self.no_temporal_pos_emb = no_temporal_pos_emb
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.enable_flashattn = enable_flashattn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        self.space_scale = space_scale
        self.time_scale = time_scale

        self.register_buffer("pos_embed", self.get_spatial_pos_embed())
        self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed())

        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )

        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                STDiTBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=drop_path[i],
                    enable_flashattn=self.enable_flashattn,
                    enable_layernorm_kernel=self.enable_layernorm_kernel,
                    d_t=self.num_temporal,
                    d_s=self.num_spatial,
                )
                for i in range(self.depth)
            ]
        )
        self.final_layer = STDiTFinalLayer(hidden_size, np.prod(self.patch_size), self.out_channels)

        # init model
        self.initialize_weights()
        self.initialize_temporal()
        if freeze is not None:
            assert freeze in ["not_temporal", "text"]
            if freeze == "not_temporal":
                self.freeze_not_temporal()
            elif freeze == "text":
                self.freeze_text()

        # sequence parallel related configs
        self.enable_sequence_parallelism = use_sequecne_parallelism()

        self.gradient_checkpointing = False

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    @staticmethod
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)

        return custom_forward

    def forward(self, x, timestep, y, mask=None):
        """
        Forward pass of PixArt.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N, 1, 120, C) tensor of class labels
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)

        # embedding
        x = self.x_embedder(x)  # (B, N, D)
        x = rearrange(x, "b (t s) d -> b t s d", t=self.num_temporal, s=self.num_spatial)
        x = x + self.pos_embed
        x = rearrange(x, "b t s d -> b (t s) d")

        # shard over the sequence dim if sp is enabled
        if self.enable_sequence_parallelism:
            x = split_sequence(x, get_sequence_parallel_group(), dim=1, grad_scale="down")

        t = self.t_embedder(timestep, dtype=x.dtype)  # (N, D)
        t0 = self.t_block(t)
        y = self.y_embedder(y, self.training)  # (N, 1, L, D)

        if mask is not None:
            if mask.shape[0] != y.shape[0]:
                mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
            mask = mask.squeeze(1).squeeze(1)
            y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
            y_lens = mask.sum(dim=1).tolist()
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        # blocks
        for i, block in enumerate(self.blocks):
            if i == 0:
                tpe = self.pos_embed_temporal
            else:
                tpe = None

            if self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(self.create_custom_forward(block), x, y, t0, y_lens, tpe)
            else:
                x = block(x, y, t0, y_lens, tpe)

        if self.enable_sequence_parallelism:
            x = gather_sequence(x, get_sequence_parallel_group(), dim=1, grad_scale="up")

        # final process
        x = self.final_layer(x, t)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        return x

    def unpatchify(self, x):
        c = self.out_channels
        t, h, w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        pt, ph, pw = self.patch_size

        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        x = rearrange(x, "n t h w r p q c -> n c t r h p w q")
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def get_spatial_pos_embed(self, grid_size=None):
        if grid_size is None:
            grid_size = self.input_size[1:]
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            (grid_size[0] // self.patch_size[1], grid_size[1] // self.patch_size[2]),
            scale=self.space_scale,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def get_temporal_pos_embed(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.hidden_size,
            self.input_size[0] // self.patch_size[0],
            scale=self.time_scale,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def freeze_not_temporal(self):
        for n, p in self.named_parameters():
            if "attn_temp" not in n:
                p.requires_grad = False

    def freeze_text(self):
        for n, p in self.named_parameters():
            if "cross_attn" in n:
                p.requires_grad = False

    def initialize_temporal(self):
        for block in self.blocks:
            nn.init.constant_(block.attn_temp.proj.weight, 0)
            nn.init.constant_(block.attn_temp.proj.bias, 0)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


def STDiT_XL_2(from_pretrained=None, **kwargs):
    model = STDiT(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
    if from_pretrained is not None:
        load_checkpoint(model, from_pretrained)
    return model
