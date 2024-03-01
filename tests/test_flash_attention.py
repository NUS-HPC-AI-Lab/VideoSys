import copy

import colossalai
import flash_attn
import pytest
import torch
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from torch.testing import assert_close

from opendit.modules.attn import DistAttention

torch.manual_seed(1024)

WORKERS = 1
DTYPE = torch.float16


def _run_flash_attn(seq_len, hidden_dim, head_num, batch_size, use_flash_attn):
    seq_len = seq_len
    hidden_dim = hidden_dim
    head_num = head_num
    batch_size = batch_size

    # set dtype as bf16
    torch.set_default_dtype(DTYPE)

    x = torch.randn(batch_size, seq_len, hidden_dim).cuda()
    x_naive_attn = x.clone().requires_grad_(True)
    x_flash_attn = x.clone().requires_grad_(True)

    # DistAttention without flash attention
    dist_attn_without_flashattn = DistAttention(
        dim=hidden_dim,
        num_heads=head_num,
        enable_flashattn=use_flash_attn,
        sequence_parallel_size=1,
        sequence_parallel_group=None,
    ).cuda()

    dist_attn_with_flashattn = copy.deepcopy(dist_attn_without_flashattn)
    setattr(dist_attn_with_flashattn, "enable_flashattn", True)

    naive_attn_output = dist_attn_without_flashattn(x_naive_attn)
    flash_attn_output = dist_attn_with_flashattn(x_flash_attn)

    assert_close(naive_attn_output, flash_attn_output, atol=1e-4, rtol=1e-4)

    # Attention backward
    naive_attn_output.sum().backward()
    qkv_grad_naive_attn = dist_attn_without_flashattn.qkv.weight.grad
    o_grad_naive_attn = dist_attn_without_flashattn.proj.weight.grad
    x_grad_naive_attn = x_naive_attn.grad

    flash_attn_output.sum().backward()
    qkv_grad_flash_attn = dist_attn_with_flashattn.qkv.weight.grad
    o_grad_flash_attn = dist_attn_with_flashattn.proj.weight.grad
    x_grad_flash_attn = x_flash_attn.grad

    # backward result check
    assert_close(qkv_grad_naive_attn, qkv_grad_flash_attn, atol=1e-3, rtol=1e-3)
    assert_close(o_grad_naive_attn, o_grad_flash_attn, atol=1e-3, rtol=1e-3)
    assert_close(x_grad_naive_attn, x_grad_flash_attn, atol=1e-3, rtol=1e-3)


@parameterize("seq_len", [256])
@parameterize("hidden_dim", [1152])
@parameterize("head_num", [16])
@parameterize("batch_size", [2])
@parameterize("use_flash_attn", [True])
def run_flash_attn(seq_len, hidden_dim, head_num, batch_size, use_flash_attn):
    _run_flash_attn(seq_len, hidden_dim, head_num, batch_size, use_flash_attn)


def check_all2all_attn(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_flash_attn()


@pytest.mark.skipif(flash_attn.__version__ < "2.4.1", reason="requires flashattn 2.4.1 or higher")
@rerun_if_address_is_in_use()
def test_flash_attn():
    spawn(check_all2all_attn, nprocs=WORKERS)


if __name__ == "__main__":
    test_flash_attn()
