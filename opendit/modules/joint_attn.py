import math
import torch
import torch.nn as nn
import numpy as np

from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp, PatchEmbed

# TODO: make attention use processors similar to attn.py
try:
    import xformers
    import xformers.ops
except:
    XFORMERS_IS_AVAILBLE = False

try:
    # needs to have https://github.com/corl-team/rebased/ installed
    from fla.ops.triton.rebased_fast import parallel_rebased
except:
    REBASED_IS_AVAILABLE = False

try:
    # needs to have https://github.com/lucidrains/ring-attention-pytorch installed
    from ring_attention_pytorch.ring_flash_attention_cuda import ring_flash_attn_cuda
except:
    RING_ATTENTION_IS_AVAILABLE = False

class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed

class JointAttention(nn.Module):
    def __init__(self, txt_dim, pix_dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_lora=False, attention_mode='math', eps=1e-12, causal=True, ring_bucket_size=1024):
        super().__init__()
        dim = txt_dim + pix_dim
        assert txt_dim % num_heads == 0, 'dim should be divisible by num_heads'
        assert pix_dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attention_mode = attention_mode
        self.qkv_text = nn.Linear(txt_dim, txt_dim * 3, bias=qkv_bias)
        self.qkv_pix = nn.Linear(pix_dim, pix_dim * 3, bias=qkv_bias)
        self.rms_q_text = RMSNorm(txt_dim)
        self.rms_k_text = RMSNorm(txt_dim)
        self.rms_q_pix = RMSNorm(pix_dim)
        self.rms_k_pix = RMSNorm(pix_dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_pix = nn.Linear(dim, txt_dim)
        self.proj_text = nn.Linear(dim, pix_dim)
        self.proj_drop_text = nn.Dropout(proj_drop)
        self.proj_drop_pix = nn.Dropout(proj_drop)
        self.eps = eps
        self.causal = causal
        self.ring_bucket_size = ring_bucket_size

    def forward(self, x, c):
        B, N, C1 = x.shape
        qkv_pix = self.qkv_pix(x).reshape(B, N, 3, self.num_heads, C1 // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q_pix, k_pix, v_pix = qkv_pix.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        q_pix = self.rms_q_pix(q_pix)
        k_pix = self.rms_k_pix(k_pix)

        # Assuming 
        B, N, C2 = c.shape
        qkv_text = self.qkv_text(x).reshape(B, N, 3, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q_text, k_text, v_text = qkv_text.unbind(0)   # make torchscript happy (cannot use tensor as tuple)

        if self.attention_mode != 'rebased':
            # Rebased does RMS norm inside already
            q_text = self.rms_q_text(q_text)
            k_text = self.rms_k_text(k_text)

        q = torch.cat([q_text, q_pix], dim=-1)
        k = torch.cat([k_text, k_pix], dim=-1)
        v = torch.cat([v_text, v_pix], dim=-1)

        C = C1 + C2

        if self.attention_mode == 'xformers': # cause loss nan while using with amp
            z = xformers.ops.memory_efficient_attention(q, k, v).reshape(B, N, C)

        elif self.attention_mode == 'flash':
            # cause loss nan while using with amp
            # Optionally use the context manager to ensure one of the fused kerenels is run
            with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=False):
                z = torch.nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p, scale=self.scale).reshape(B, N, C) # require pytorch 2.0

        elif self.attention_mode == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            z = (attn @ v).transpose(1, 2).reshape(B, N, C)

        elif self.attention_mode == 'rebased':
            z = parallel_rebased(q, k, v, self.eps, True, True).reshape(B, N, C)

        elif self.attention_mode == 'ring':
            z = ring_flash_attn_cuda(q, k, v, causal=self.causal, bucket_size=self.ring_bucket_size).reshape(B, N, C)

        else:
            raise NotImplemented

        x = self.proj_pix(z)
        x = self.proj_drop_pix(x)

        c = self.proj_text(z)
        c = self.proj_drop_text(c)

        return x, c
