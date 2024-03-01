import copy

import colossalai
import torch
import torch.distributed as dist
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from torch.testing import assert_close

from opendit.modules.attn import DistAttention
from opendit.modules.block import DiTBlock

WORKERS = 4
DTYPE = torch.float16


def seq_parallel_attn(seq_len, hidden_dim, head_num, batch_size, use_flash_attn):
    seq_len = seq_len
    hidden_dim = hidden_dim
    head_num = head_num
    batch_size = batch_size
    world_size = dist.get_world_size()

    torch.set_default_dtype(DTYPE)

    x = torch.randn(batch_size, seq_len, hidden_dim).cuda()
    x_unshard = x.clone()
    x_unshard.requires_grad_(True)
    x_shard = torch.chunk(x.clone(), world_size, dim=1)[dist.get_rank()]
    x_shard.requires_grad_(True)

    # DistAttention without sequence parallel
    dist_attn_without_sp = DistAttention(
        dim=hidden_dim,
        num_heads=head_num,
        enable_flashattn=use_flash_attn,
        sequence_parallel_size=1,
        sequence_parallel_group=None,
    ).cuda()

    # DistAttention with sequence parallel
    dist_attn_with_sp = copy.deepcopy(dist_attn_without_sp)
    setattr(dist_attn_with_sp, "sequence_parallel_size", world_size)
    setattr(dist_attn_with_sp, "sequence_parallel_group", None)

    # Attention forward (Without Sequence parallel)
    no_sp_output = dist_attn_without_sp(x_unshard)

    # Attention forward (With Sequence parallel)
    sp_output = dist_attn_with_sp(x_shard)
    # gather the output of sequence parallel attention
    out_list = [torch.empty_like(sp_output) for _ in range(world_size)]
    dist.all_gather(out_list, sp_output)
    seq_out = torch.cat(out_list, dim=1)

    # forward result check
    assert_close(seq_out, no_sp_output, atol=1e-4, rtol=1e-4)

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
    assert_close(qkv_grad_sp, qkv_grad_no_sp, atol=5e-2, rtol=1e-2)
    assert_close(o_grad_sp, o_grad_no_sp, atol=5e-3, rtol=5e-3)
    assert_close(x_grad_seq_gather, x_unshard_grad, atol=1e-3, rtol=1e-3)


def seq_parallel_block(seq_len, hidden_dim, head_num, batch_size, use_flash_attn, layernorm_kernel, modulate_kernel):
    seq_len = seq_len
    hidden_dim = hidden_dim
    head_num = head_num
    batch_size = batch_size
    world_size = dist.get_world_size()

    torch.set_default_dtype(DTYPE)

    x = torch.randn(batch_size, seq_len, hidden_dim).cuda()
    x_unshard = x.clone()
    x_unshard.requires_grad_(True)
    x_shard = torch.chunk(x.clone(), world_size, dim=1)[dist.get_rank()]
    x_shard.requires_grad_(True)

    c = torch.randn(batch_size, hidden_dim).cuda()
    c_unshard = c.clone()
    c_unshard.requires_grad_(True)
    c_shard = c.clone()
    c_shard.requires_grad_(True)

    # DistAttention without sequence parallel
    dist_block_without_sp = DiTBlock(
        hidden_size=hidden_dim,
        num_heads=head_num,
        enable_flashattn=use_flash_attn,
        sequence_parallel_size=1,
        sequence_parallel_group=None,
        enable_layernorm_kernel=layernorm_kernel,
        enable_modulate_kernel=modulate_kernel,
    ).cuda()

    # DistAttention with sequence parallel
    dist_block_with_sp = copy.deepcopy(dist_block_without_sp)
    setattr(dist_block_with_sp, "sequence_parallel_size", world_size)
    setattr(dist_block_with_sp, "sequence_parallel_group", None)
    setattr(dist_block_with_sp.attn, "sequence_parallel_size", world_size)
    setattr(dist_block_with_sp.attn, "sequence_parallel_group", None)

    # Attention forward (Without Sequence parallel)
    no_sp_output = dist_block_without_sp(x_unshard, c_unshard)

    # Attention forward (With Sequence parallel)
    sp_output = dist_block_with_sp(x_shard, c_shard)
    # gather the output of sequence parallel attention
    out_list = [torch.empty_like(sp_output) for _ in range(world_size)]
    dist.all_gather(out_list, sp_output.contiguous())
    seq_out = torch.cat(out_list, dim=1)

    # forward result check
    assert_close(seq_out, no_sp_output, atol=1e-4, rtol=1e-4)

    # Attention backward (Without Sequence parallel)
    no_sp_output.sum().backward()
    x_unshard_grad = x_unshard.grad

    # Attention backward (With Sequence parallel)
    sp_output.sum().backward()
    x_shard_grad = x_shard.grad

    # gather the grad of sequence parallel attention input
    x_grad_seq_list = [torch.empty_like(x_shard_grad) for _ in range(world_size)]
    dist.all_gather(x_grad_seq_list, x_shard_grad)
    x_grad_seq_gather = torch.cat(x_grad_seq_list, dim=1)

    # backward result check
    assert_close(x_grad_seq_gather, x_unshard_grad, atol=1e-3, rtol=1e-3)


@parameterize("seq_len", [256])
@parameterize("hidden_dim", [1152])
@parameterize("head_num", [16])
@parameterize("batch_size", [2])
@parameterize("use_flash_attn", [True, False])
def run_seq_parallel_attn(seq_len, hidden_dim, head_num, batch_size, use_flash_attn):
    seq_parallel_attn(seq_len, hidden_dim, head_num, batch_size, use_flash_attn)


@parameterize("seq_len", [256])
@parameterize("hidden_dim", [1152])
@parameterize("head_num", [16])
@parameterize("batch_size", [2])
@parameterize("use_flash_attn", [False])
@parameterize("layernorm_kernel", [True])
@parameterize("modulate_kernel", [True])
def run_seq_parallel_block(
    seq_len, hidden_dim, head_num, batch_size, use_flash_attn, layernorm_kernel, modulate_kernel
):
    seq_parallel_block(seq_len, hidden_dim, head_num, batch_size, use_flash_attn, layernorm_kernel, modulate_kernel)


def check_all2all_attn(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_seq_parallel_attn()


@rerun_if_address_is_in_use()
def test_sequence_parallel_attn():
    spawn(check_all2all_attn, nprocs=WORKERS)


def check_all2all_block(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_seq_parallel_block()


@rerun_if_address_is_in_use()
def test_sequence_parallel_block():
    spawn(check_all2all_block, nprocs=WORKERS)


if __name__ == "__main__":
    test_sequence_parallel_attn()
    test_sequence_parallel_block()
