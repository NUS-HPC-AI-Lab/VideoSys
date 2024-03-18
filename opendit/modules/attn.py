from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from torch.distributed import ProcessGroup

from opendit.utils.operation import AllGather, AsyncAllGatherForTwo, all_to_all_comm


class DistAttention(nn.Module):
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
        if self.sequence_parallel_size > 1:
            assert (
                self.num_heads % self.sequence_parallel_size == 0
            ), "num_heads should be divisible by sequence_parallel_size"
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

    # Rearrange the qkv projection (qkv ... qkv <-> q ... q k ... k v ... v)
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


class Attention(nn.Module):
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
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flashattn = enable_flashattn

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
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
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        x_output_shape = (B, N, C)
        if not self.enable_flashattn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads, attn_drop=0.0, proj_drop=0.0, enable_flashattn=False):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.enable_flashattn = enable_flashattn

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, d_model * 2)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(d_model, d_model)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, cond, mask=None):
        # query/value: img tokens; key: condition; mask: if padding tokens
        B, N, C = x.shape

        q = self.q_linear(x).view(1, -1, self.num_heads, self.head_dim)
        kv = self.kv_linear(cond).view(1, -1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)

        # import xformers
        # import xformers.ops
        # attn_bias = None
        # if mask is not None:
        #     attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens([N] * B, mask)
        # x = xformers.ops.memory_efficient_attention(q, k, v, p=self.attn_drop.p, attn_bias=attn_bias)
        # x = x.view(B, -1, C)

        if self.enable_flashattn:
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

        attn_mask = torch.zeros(B, N, k.shape[2], dtype=torch.bool, device=q.device)
        for i, m in enumerate(mask):
            attn_mask[i, :, :m] = -1e9

        scale = 1 / q.shape[-1] ** 0.5
        q = q * scale
        attn = q @ k.transpose(-2, -1)
        if mask is not None:
            attn = attn + attn_mask.unsqueeze(1)
        attn = attn.to(torch.float32)
        attn = attn.softmax(-1)
        attn = attn.to(v.dtype)
        out = attn @ v

        x = out.transpose(1, 2).contiguous().view(B, N, C)
        return x


@dataclass
class _SeqLenInfo:
    """
    copied from xformers

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

    def split(self, x: torch.Tensor, batch_sizes: Optional[Sequence[int]] = None) -> List[torch.Tensor]:
        if self.seqstart_py[-1] != x.shape[1] or x.shape[0] != 1:
            raise ValueError(
                f"Invalid `torch.Tensor` of shape {x.shape}, expected format "
                f"(B, M, *) with B=1 and M={self.seqstart_py[-1]}\n"
                f" seqstart: {self.seqstart_py}"
            )
        if batch_sizes is None:
            batch_sizes = [1] * (len(self.seqstart_py) - 1)
        split_chunks = []
        it = 0
        for batch_size in batch_sizes:
            split_chunks.append(self.seqstart_py[it + batch_size] - self.seqstart_py[it])
            it += batch_size
        return [
            tensor.reshape([bs, -1, *tensor.shape[2:]]) for bs, tensor in zip(batch_sizes, x.split(split_chunks, dim=1))
        ]
