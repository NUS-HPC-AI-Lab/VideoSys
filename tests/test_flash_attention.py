import copy

import colossalai
import torch
import torch.nn as nn
import torch.nn.functional as F
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from timm.models.vision_transformer import use_fused_attn
from torch.jit import Final
from torch.testing import assert_close

from opendit.utils.operation import all_to_all_comm

torch.manual_seed(1024)

WORKERS = 1
DTYPE = torch.float16


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
        fused_attn: bool = use_fused_attn(),
        use_flash_attn: bool = True,
        enable_sequence_parallelism: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = fused_attn

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
                # [3, B, num_heads, N, head_dim] => [B, N, num_heads, head_dim] * 3
                qkv = qkv.reshape(B, N, 3, num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
            else:
                # [3, B, num_heads, N, head_dim] => [B, num_heads, N, head_dim] * 3
                qkv = qkv.reshape(B, N, 3, num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            # qkv = qkv.reshape(B, N, 3, num_heads, self.head_dim).permute(2, 0, 1, 3, 4)
            # q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.use_flash_attn:
            from flash_attn import flash_attn_func

            x = flash_attn_func(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0, deterministic=True)

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
            attn.to(torch.float32)
            attn = attn.softmax(dim=-1)
            attn.to(DTYPE)
            attn = self.attn_drop(attn)
            x = attn @ v

        x_output_shape = (
            (B, N, C) if not self.enable_sequence_parallelism else (B, N * WORKERS, num_heads * self.head_dim)
        )
        if self.use_flash_attn:
            x = x.reshape(x_output_shape)
        else:
            x = x.transpose(1, 2).reshape(x_output_shape)
        if self.enable_sequence_parallelism:
            # Todo: Use all_to_all_single for x
            # x = x.reshape(1, -1, num_heads * self.head_dim)
            x = all_to_all_comm(x, None, scatter_dim=1, gather_dim=2)
            # x = x.reshape(B, -1, num_heads * self.head_dim * SP_SIZE)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def flash_attn(seq_len, hidden_dim, head_num, batch_size):
    seq_len = seq_len
    hidden_dim = hidden_dim
    head_num = head_num
    batch_size = batch_size

    # set dtype as bf16
    torch.set_default_dtype(DTYPE)

    qkv_proj_naive_attn = nn.Linear(hidden_dim, 3 * hidden_dim)
    out_proj_naive_attn = nn.Linear(hidden_dim, hidden_dim)

    qkv_proj_fused_attn = copy.deepcopy(qkv_proj_naive_attn)
    out_proj_fused_attn = copy.deepcopy(out_proj_naive_attn)

    qkv_proj_flash_attn = copy.deepcopy(qkv_proj_naive_attn)
    out_proj_flash_attn = copy.deepcopy(out_proj_naive_attn)

    x = torch.randn(batch_size, seq_len, hidden_dim).cuda()
    x_naive_attn = x.clone().requires_grad_(True)
    x_fused_attn = x.clone().requires_grad_(True)
    x_flash_attn = x.clone().requires_grad_(True)

    # DistAttention: vanilla attention
    dist_naive_attn = DistAttention(
        dim=hidden_dim,
        num_heads=head_num,
        qkv_proj=qkv_proj_naive_attn,
        output_proj=out_proj_naive_attn,
        fused_attn=False,
        use_flash_attn=False,
        enable_sequence_parallelism=False,
    ).cuda()

    naive_attn_output = dist_naive_attn(x_naive_attn)

    # DistAttention: fused attention
    dist_fused_attn = DistAttention(
        dim=hidden_dim,
        num_heads=head_num,
        qkv_proj=qkv_proj_fused_attn,
        output_proj=out_proj_fused_attn,
        use_flash_attn=False,
        enable_sequence_parallelism=False,
    ).cuda()

    fused_attn_output = dist_fused_attn(x_fused_attn)

    # DistAttention: flash attention
    dist_flash_attn = DistAttention(
        dim=hidden_dim,
        num_heads=head_num,
        qkv_proj=qkv_proj_flash_attn,
        output_proj=out_proj_flash_attn,
        use_flash_attn=True,
        enable_sequence_parallelism=False,
    ).cuda()

    flash_attn_output = dist_flash_attn(x_flash_attn)

    # forward result check
    # assert torch.allclose(
    #     naive_attn_output, fused_attn_output, atol=1e-4, rtol=1e-4
    # ), "difference between naive and fused attention (forward)"
    # print_rank("naive_attn_output", naive_attn_output)
    # print_rank("flash_attn_output", flash_attn_output)

    assert_close(naive_attn_output, flash_attn_output, atol=1e-4, rtol=1e-4)
    # assert torch.allclose(flash_attn_output, fused_attn_output, atol=1e-4, rtol=1e-4), "difference between fused and flash attention (forward)"

    # Attention backward
    naive_attn_output.sum().backward()
    qkv_grad_naive_attn = dist_naive_attn.qkv.weight.grad
    o_grad_naive_attn = dist_naive_attn.proj.weight.grad
    x_grad_naive_attn = x_naive_attn.grad

    fused_attn_output.sum().backward()
    dist_fused_attn.qkv.weight.grad
    dist_fused_attn.proj.weight.grad
    x_fused_attn.grad

    flash_attn_output.sum().backward()
    qkv_grad_flash_attn = dist_flash_attn.qkv.weight.grad
    o_grad_flash_attn = dist_flash_attn.proj.weight.grad
    x_grad_flash_attn = x_flash_attn.grad

    # backward result check
    # assert torch.allclose(
    #     qkv_grad_naive_attn, qkv_grad_fused_attn, atol=1e-4, rtol=1e-4
    # ), "difference between naive and fused attention (backward, qkv)"
    # assert torch.allclose(
    #     o_grad_naive_attn, o_grad_fused_attn, atol=1e-4, rtol=1e-4
    # ), "difference between naive and fused attention (backward, o)"
    # assert torch.allclose(
    #     x_grad_naive_attn, x_grad_fused_attn, atol=1e-4, rtol=1e-4
    # ), "difference between naive and fused attention (backward, x)"

    assert_close(qkv_grad_naive_attn, qkv_grad_flash_attn, atol=1e-3, rtol=1e-3)
    assert_close(o_grad_naive_attn, o_grad_flash_attn, atol=1e-3, rtol=1e-3)
    assert_close(x_grad_naive_attn, x_grad_flash_attn, atol=1e-3, rtol=1e-3)

    # assert torch.allclose(qkv_grad_fused_attn, qkv_grad_flash_attn, atol=1e-4, rtol=1e-4), "difference between fused and flash attention (backward, qkv)"
    # assert torch.allclose(o_grad_fused_attn, o_grad_flash_attn, atol=1e-4, rtol=1e-4), "difference between fused and flash attention (backward, o)"
    # assert torch.allclose(x_grad_fused_attn, x_grad_flash_attn, atol=1e-4, rtol=1e-4), "difference between fused and flash attention (backward, x)"


@parameterize("seq_len", [256])
@parameterize("hidden_dim", [1152])
@parameterize("head_num", [16])
@parameterize("batch_size", [2])
def run_flash_attn(seq_len, hidden_dim, head_num, batch_size):
    flash_attn(seq_len, hidden_dim, head_num, batch_size)


def check_all2all_attn(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_flash_attn()


@rerun_if_address_is_in_use()
def test_flash_attn():
    spawn(check_all2all_attn, nprocs=WORKERS)


if __name__ == "__main__":
    test_flash_attn()
