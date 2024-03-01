from typing import Optional

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
