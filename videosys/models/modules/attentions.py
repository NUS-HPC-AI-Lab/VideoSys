from dataclasses import dataclass
from typing import Iterable, List, Tuple
import inspect
import math
from importlib import import_module
from typing import Callable, List, Optional, Union
from diffusers.models.attention import Attention

import torch
import torch.nn.functional as F
from torch import nn

from diffusers.image_processor import IPAdapterMaskProcessor
from diffusers.utils import deprecate, logging
from diffusers.utils.import_utils import is_torch_npu_available, is_xformers_available
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.lora import LoRALinearLayer
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

from videosys.models.modules.normalization import LlamaRMSNorm


class OpenSoraAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = LlamaRMSNorm,
        enable_flash_attn: bool = False,
        rope=None,
        qk_norm_legacy: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flash_attn = enable_flash_attn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.qk_norm_legacy = qk_norm_legacy
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.rope = False
        if rope is not None:
            self.rope = True
            self.rotary_emb = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # flash attn is not memory efficient for small sequences, this is empirical
        enable_flash_attn = self.enable_flash_attn and (N > B)
        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)

        qkv = qkv.view(qkv_shape).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.qk_norm_legacy:
            # WARNING: this may be a bug
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)
            q, k = self.q_norm(q), self.k_norm(k)
        else:
            q, k = self.q_norm(q), self.k_norm(k)
            if self.rope:
                q = self.rotary_emb(q)
                k = self.rotary_emb(k)

        if enable_flash_attn:
            from flash_attn import flash_attn_func

            # (B, #heads, N, #dim) -> (B, N, #heads, #dim)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
            )
        else:
            x = F.scaled_dot_product_attention(q, k, v)

        x_output_shape = (B, N, C)
        if not enable_flash_attn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class OpenSoraMultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, enable_flash_attn=False):
        super(OpenSoraMultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)
        self.enable_flash_attn = enable_flash_attn

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        if self.enable_flash_attn:
            x = self.flash_attn_impl(q, k, v, mask, B, N, C)
        else:
            x = self.torch_impl(q, k, v, mask, B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def flash_attn_impl(self, q, k, v, mask, B, N, C):
        from flash_attn import flash_attn_varlen_func

        q_seqinfo = _SeqLenInfo.from_seqlens([N] * B)
        k_seqinfo = _SeqLenInfo.from_seqlens(mask)

        x = flash_attn_varlen_func(
            q.view(-1, self.num_heads, self.head_dim),
            k.view(-1, self.num_heads, self.head_dim),
            v.view(-1, self.num_heads, self.head_dim),
            cu_seqlens_q=q_seqinfo.seqstart.cuda(),
            cu_seqlens_k=k_seqinfo.seqstart.cuda(),
            max_seqlen_q=q_seqinfo.max_seqlen,
            max_seqlen_k=k_seqinfo.max_seqlen,
            dropout_p=self.attn_drop.p if self.training else 0.0,
        )
        x = x.view(B, N, C)
        return x

    def torch_impl(self, q, k, v, mask, B, N, C):
        q = q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn_mask = torch.zeros(B, 1, N, k.shape[2], dtype=torch.bool, device=q.device)
        for i, m in enumerate(mask):
            attn_mask[i, :, :, :m] = -1e9

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        x = out.transpose(1, 2).contiguous().view(B, N, C)
        return x


@dataclass
class _SeqLenInfo:
    """
    from xformers

    (Internal) Represents the division of a dimension into blocks.
    For example, to represents a dimension of length 7 divided into
    three blocks of lengths 2, 3 and 2, use `from_seqlength([2, 3, 2])`.
    The members will be:
        max_seqlen: 3
        min_seqlen: 2
        seqstart_py: [0, 2, 5, 7]
        seqstart: torch.IntTensor([0, 2, 5, 7])
    """

    seqstart: torch.Tensor
    max_seqlen: int
    min_seqlen: int
    seqstart_py: List[int]

    def to(self, device: torch.device) -> None:
        self.seqstart = self.seqstart.to(device, non_blocking=True)

    def intervals(self) -> Iterable[Tuple[int, int]]:
        yield from zip(self.seqstart_py, self.seqstart_py[1:])

    @classmethod
    def from_seqlens(cls, seqlens: Iterable[int]) -> "_SeqLenInfo":
        """
        Input tensors are assumed to be in shape [B, M, *]
        """
        assert not isinstance(seqlens, torch.Tensor)
        seqstart_py = [0]
        max_seqlen = -1
        min_seqlen = -1
        for seqlen in seqlens:
            min_seqlen = min(min_seqlen, seqlen) if min_seqlen != -1 else seqlen
            max_seqlen = max(max_seqlen, seqlen)
            seqstart_py.append(seqstart_py[len(seqstart_py) - 1] + seqlen)
        seqstart = torch.tensor(seqstart_py, dtype=torch.int32)
        return cls(
            max_seqlen=max_seqlen,
            min_seqlen=min_seqlen,
            seqstart=seqstart,
            seqstart_py=seqstart_py,
        )


class VchitectAttnProcessor:

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def reshape_for_broadcast(self, freqs_cis: torch.Tensor, x: torch.Tensor):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs_cis.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1
                 for i, d in enumerate(x.shape)]
        return freqs_cis.view(*shape)

    def apply_rotary_emb(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor,
    ):
        with torch.cuda.amp.autocast(enabled=False):
            xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
            xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
            freqs_cis = self.reshape_for_broadcast(freqs_cis, xq_)
            xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
            xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
            return xq_out.type_as(xq), xk_out.type_as(xk)

        
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
        full_seqlen: Optional[int] = None,
        Frame: Optional[int] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size = encoder_hidden_states.shape[0]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query_t = attn.to_q_temp(hidden_states)
        key_t = attn.to_k_temp(hidden_states)
        value_t = attn.to_v_temp(hidden_states)

        query_cross = attn.to_q_cross(hidden_states)

        # `context` projections.
        encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
        encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
        encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)
        

        query_cross = torch.cat([query_cross, encoder_hidden_states_query_proj], dim=1)
        # attention
        query = torch.cat([query, encoder_hidden_states_query_proj], dim=1)
        key = torch.cat([key, encoder_hidden_states_key_proj], dim=1)
        value = torch.cat([value, encoder_hidden_states_value_proj], dim=1)

        query_t = torch.cat([query_t, encoder_hidden_states_query_proj], dim=1)
        key_t = torch.cat([key_t, encoder_hidden_states_key_proj], dim=1)
        value_t = torch.cat([value_t, encoder_hidden_states_value_proj], dim=1)

        key_y = encoder_hidden_states_key_proj[0].unsqueeze(0)
        value_y = encoder_hidden_states_value_proj[0].unsqueeze(0)


        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        batchsize = query.shape[0] // Frame

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)

        query_t = query_t.view(batch_size, -1, attn.heads, head_dim)
        key_t = key_t.view(batch_size, -1, attn.heads, head_dim)
        value_t = value_t.view(batch_size, -1, attn.heads, head_dim)

        query_y = query_cross.view(batch_size, -1, attn.heads, head_dim)
        key_y = key_y.view(batchsize, -1, attn.heads, head_dim)
        value_y = value_y.view(batchsize, -1, attn.heads, head_dim)

        # temporal
        xq, xk = query.to(value.dtype), key.to(value.dtype)

        xq_gather = attn.gather_seq_scatter_hidden(xq)
        xk_gather = attn.gather_seq_scatter_hidden(xk)
        xv_gather = attn.gather_seq_scatter_hidden(value)

        xq_t, xk_t = query_t.to(value_t.dtype), key_t.to(value_t.dtype)

        xq_gather_t = attn.gather_seq_scatter_hidden(xq_t)
        xk_gather_t = attn.gather_seq_scatter_hidden(xk_t)
        xv_gather_t = attn.gather_seq_scatter_hidden(value_t)

        query_spatial, key_spatial, value_spatial = xq_gather.clone(), xk_gather.clone(), xv_gather.clone()

        xq_gather_temporal = rearrange(xq_gather_t, "(B T) S H C -> (B S) T H C", T=Frame, B=batchsize)
        xk_gather_temporal = rearrange(xk_gather_t, "(B T) S H C -> (B S) T H C", T=Frame, B=batchsize)
        xv_gather_temporal = rearrange(xv_gather_t, "(B T) S H C -> (B S) T H C", T=Frame, B=batchsize)

        freqs_cis_temporal = freqs_cis[:xq_gather_temporal.shape[1],:]
        xq_gather_temporal, xk_gather_temporal = self.apply_rotary_emb(xq_gather_temporal, xk_gather_temporal, freqs_cis=freqs_cis_temporal)

        query_spatial = query_spatial.transpose(1, 2)
        key_spatial = key_spatial.transpose(1, 2)
        value_spatial = value_spatial.transpose(1, 2)

        xq_gather_temporal = xq_gather_temporal.transpose(1, 2)
        xk_gather_temporal = xk_gather_temporal.transpose(1, 2)
        xv_gather_temporal = xv_gather_temporal.transpose(1, 2)

        batch_size_temp = xv_gather_temporal.shape[0]
        hidden_states_temp = hidden_states_temp = F.scaled_dot_product_attention(
            xq_gather_temporal, xk_gather_temporal, xv_gather_temporal, dropout_p=0.0, is_causal=False
        )
        hidden_states_temp = hidden_states_temp.transpose(1, 2).reshape(batch_size_temp, -1, attn.heads * head_dim)
        hidden_states_temp = hidden_states_temp.to(query.dtype)
        hidden_states_temp = rearrange(hidden_states_temp, "(B S) T C -> (B T) S C", T=Frame, B=batchsize)
        #######
        
        hidden_states = hidden_states = F.scaled_dot_product_attention(
            query_spatial, key_spatial, value_spatial, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.gather_hidden_scatter_seq(hidden_states)
        hidden_states_temp = attn.gather_hidden_scatter_seq(hidden_states_temp)

        query_y = rearrange(query_y, "(B T) S H C -> B (S T) H C", T=Frame, B=batchsize)

        query_y = query_y.transpose(1, 2)
        key_y = key_y.transpose(1, 2)
        value_y = value_y.transpose(1, 2)

        cross_output = F.scaled_dot_product_attention(
            query_y, key_y, value_y, dropout_p=0.0, is_causal=False
        )
        cross_output = cross_output.transpose(1, 2).reshape(batchsize, -1, attn.heads * head_dim)
        cross_output = cross_output.to(query.dtype)

        cross_output = rearrange(cross_output, "B (S T) C -> (B T) S C", T=Frame, B=batchsize)
        cross_output = attn.to_out_context(cross_output)

        hidden_states = hidden_states*1.1 + cross_output

        # Split the attention outputs.
        hidden_states, encoder_hidden_states = (
            hidden_states[:, : residual.shape[1]],
            hidden_states[:, residual.shape[1] :],
        )
        hidden_states_temporal, encoder_hidden_states_temporal = (
            hidden_states_temp[:, : residual.shape[1]],
            hidden_states_temp[:, residual.shape[1] :],
        )
        hidden_states_temporal = attn.to_out_temporal(hidden_states_temporal)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)
        if Frame == 1:
            hidden_states_temporal = hidden_states_temporal * 0
        hidden_states = hidden_states + hidden_states_temporal
        
        if not attn.context_pre_only:
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
        encoder_hidden_states_temporal = attn.to_add_out_temporal(encoder_hidden_states_temporal)
        if Frame == 1:
            encoder_hidden_states_temporal = encoder_hidden_states_temporal * 0
        encoder_hidden_states = encoder_hidden_states + encoder_hidden_states_temporal

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states = encoder_hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        return hidden_states, encoder_hidden_states
