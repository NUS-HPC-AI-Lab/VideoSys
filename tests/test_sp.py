import copy

import colossalai
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# from opendit.utils.operation import all_to_all_comm
from colossalai.shardformer.layer import all_to_all_comm
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from flash_attn import flash_attn_func
from timm.models.vision_transformer import use_fused_attn
from torch.jit import Final
from torch.testing import assert_close

WORKERS = 4


class DistAttention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_proj: nn.Linear = None,
        output_proj: nn.Linear = None,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        use_flash_attn: bool = True,
        enable_sequence_parallelism: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.qkv = qkv_proj
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = output_proj
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_flash_attn = use_flash_attn
        self.enable_sequence_parallelism = enable_sequence_parallelism

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x)  # (B, N, C), N here is N_total // SP_SIZE
        # Todo: Change num_heads in somewhere else for a better code style
        num_heads = self.num_heads if not self.enable_sequence_parallelism else self.num_heads // WORKERS

        if self.enable_sequence_parallelism:
            q, k, v = qkv.split(self.head_dim * self.num_heads, dim=-1)
            # q = q.reshape(1, -1, self.head_dim * self.num_heads)
            # k = k.reshape(1, -1, self.head_dim * self.num_heads)
            # v = v.reshape(1, -1, self.head_dim * self.num_heads)

            q = all_to_all_comm(q, None)
            k = all_to_all_comm(k, None)
            v = all_to_all_comm(v, None)

            q = q.reshape(B, N * WORKERS, num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
            k = k.reshape(B, N * WORKERS, num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
            v = v.reshape(B, N * WORKERS, num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()

        else:
            if self.use_flash_attn:
                # [B, N, 3, num_heads, head_dim] => [3, B * num_heads, 1, N, head_dim]
                qkv = (
                    qkv.reshape(B, N, 3, num_heads, self.head_dim)
                    .permute(2, 3, 0, 1, 4)
                    .reshape(3, B * num_heads, 1, N, self.head_dim)
                )
            else:
                qkv = qkv.reshape(B, N, 3, num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            # [3, B, num_heads, N, head_dim] => [B, num_heads, N, head_dim]
            q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.use_flash_attn:
            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        elif self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x_output_shape = (
            (B, N, C) if not self.enable_sequence_parallelism else (B, N * WORKERS, num_heads * self.head_dim)
        )
        x = x.transpose(1, 2).reshape(x_output_shape)
        if self.enable_sequence_parallelism:
            # Todo: Use all_to_all_single for x
            # x = x.reshape(1, -1, num_heads * self.head_dim)
            x = all_to_all_comm(x, None, scatter_dim=1, gather_dim=2)
            # x = x.reshape(B, -1, num_heads * self.head_dim * SP_SIZE)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def seq_parallel_attn(seq_len, hidden_dim, head_num, batch_size):
    seq_len = seq_len
    hidden_dim = hidden_dim
    head_num = head_num
    batch_size = batch_size
    world_size = dist.get_world_size()

    qkv_proj = nn.Linear(hidden_dim, 3 * hidden_dim)
    out_proj = nn.Linear(hidden_dim, hidden_dim)

    qkv_proj_copy = copy.deepcopy(qkv_proj)
    out_proj_copy = copy.deepcopy(out_proj)

    x = torch.randn(batch_size, seq_len, hidden_dim).cuda()
    x_unshard = x.clone()
    x_unshard.requires_grad_(True)
    x_shard = torch.chunk(x.clone(), world_size, dim=1)[dist.get_rank()]
    x_shard.requires_grad_(True)

    # DistAttention without sequence parallel
    dist_attn_without_sp = DistAttention(
        dim=hidden_dim,
        num_heads=head_num,
        qkv_proj=qkv_proj,
        output_proj=out_proj,
        use_flash_attn=False,
        enable_sequence_parallelism=False,
    ).cuda()

    # Attention forward (Without Sequence parallel)
    no_sp_output = dist_attn_without_sp(x_unshard)

    # DistAttention with sequence parallel
    dist_attn_with_sp = DistAttention(
        dim=hidden_dim,
        num_heads=head_num,
        qkv_proj=qkv_proj_copy,
        output_proj=out_proj_copy,
        use_flash_attn=False,
        enable_sequence_parallelism=True,
    ).cuda()

    # Attention forward (With Sequence parallel)
    sp_output = dist_attn_with_sp(x_shard)
    # gather the output of sequence parallel attention
    out_list = [torch.empty_like(sp_output) for _ in range(world_size)]
    dist.all_gather(out_list, sp_output)
    seq_out = torch.cat(out_list, dim=1)

    # forward result check
    assert_close(seq_out, no_sp_output)

    # Attention backward (Without Sequence parallel)
    no_sp_output.sum().backward()
    qkv_grad_no_sp = dist_attn_without_sp.qkv.weight.grad
    o_grad_no_sp = dist_attn_without_sp.proj.weight.grad
    x_unshard_grad = x_unshard.grad

    # Attention backward (With Sequence parallel)
    sp_output.sum().backward()
    qkv_grad_sp = dist_attn_with_sp.qkv.weight.grad
    o_grad_sp = dist_attn_with_sp.proj.weight.grad
    x_shard_grad = x_shard.grad
    # all_reduce the grad of sequence parallel attention weight
    dist.all_reduce(qkv_grad_sp)
    dist.all_reduce(o_grad_sp)
    # gather the grad of sequence parallel attention input
    x_grad_seq_list = [torch.empty_like(x_shard_grad) for _ in range(world_size)]
    dist.all_gather(x_grad_seq_list, x_shard_grad)
    x_grad_seq_gather = torch.cat(x_grad_seq_list, dim=1)

    # backward result check
    assert_close(qkv_grad_sp, qkv_grad_no_sp, atol=1e-4, rtol=1e-4)
    assert_close(o_grad_sp, o_grad_no_sp, atol=1e-4, rtol=1e-4)
    assert_close(x_grad_seq_gather, x_unshard_grad, atol=1e-4, rtol=1e-4)


@parameterize("seq_len", [128])
@parameterize("hidden_dim", [1024])
@parameterize("head_num", [16])
@parameterize("batch_size", [4])
def run_seq_parallel_attn(seq_len, hidden_dim, head_num, batch_size):
    seq_parallel_attn(seq_len, hidden_dim, head_num, batch_size)


def check_all2all_attn(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_seq_parallel_attn()


@rerun_if_address_is_in_use()
def test_sequence_parallel():
    spawn(check_all2all_attn, nprocs=WORKERS)


if __name__ == "__main__":
    test_sequence_parallel()
