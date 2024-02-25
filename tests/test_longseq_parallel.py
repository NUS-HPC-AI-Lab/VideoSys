import copy

import colossalai
import torch
import torch.distributed as dist
from colossalai.testing import parameterize, rerun_if_address_is_in_use, spawn
from torch.testing import assert_close

from opendit.models.dit import DistAttention

WORKERS = 4
DTYPE = torch.float16


def seq_parallel_attn(
    seq_len, hidden_dim, head_num, batch_size, sequence_parallel_type, sequence_parallel_overlap, use_flash_attn
):
    seq_len = seq_len
    hidden_dim = hidden_dim
    head_num = head_num
    batch_size = batch_size
    world_size = dist.get_world_size()
    if use_flash_attn:
        torch.set_default_dtype(DTYPE)
    else:
        torch.set_default_dtype(torch.float32)

    x = torch.randn(batch_size, seq_len, hidden_dim).cuda()
    x_unshard = x.clone()
    x_unshard.requires_grad_(True)
    x_shard = torch.chunk(x.clone(), world_size, dim=1)[dist.get_rank()]
    x_shard.requires_grad_(True)

    # DistAttention without sequence parallel
    dist_attn_without_sp = DistAttention(
        dim=hidden_dim,
        num_heads=head_num,
        qkv_bias=True,
        enable_flashattn=use_flash_attn,
        sequence_parallel_size=1,
        sequence_parallel_group=None,
    ).cuda()

    # DistAttention with sequence parallel
    dist_attn_with_sp = copy.deepcopy(dist_attn_without_sp)
    setattr(dist_attn_with_sp, "sequence_parallel_size", world_size)
    setattr(dist_attn_with_sp, "sequence_parallel_group", None)
    setattr(dist_attn_with_sp, "sequence_parallel_type", sequence_parallel_type)
    setattr(dist_attn_with_sp, "sequence_parallel_overlap", sequence_parallel_overlap)

    # Attention forward (Without Sequence parallel)
    no_sp_output = dist_attn_without_sp(x_unshard.contiguous())

    # Attention forward (With Sequence parallel)
    sp_output = dist_attn_with_sp(x_shard.contiguous())

    # gather the output of sequence parallel attention
    out_list = [torch.empty_like(sp_output) for _ in range(world_size)]
    dist.all_gather(out_list, sp_output)
    seq_out = torch.cat(out_list, dim=1)

    # Forward result check
    if use_flash_attn:
        assert_close(seq_out, no_sp_output, atol=1e-4, rtol=1e-4)
    else:
        # if dist.get_rank() == 0:
        # print('seq_out', seq_out)
        # print('no_sp_output', no_sp_output)

        assert_close(seq_out, no_sp_output, atol=5e-5, rtol=5e-5)

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
    # All_reduce the grad of sequence parallel attention weight
    dist.all_reduce(qkv_grad_sp)
    dist.all_reduce(o_grad_sp)
    # Gather the grad of sequence parallel attention input
    x_grad_seq_list = [torch.empty_like(x_shard_grad) for _ in range(world_size)]
    dist.all_gather(x_grad_seq_list, x_shard_grad)
    x_grad_seq_gather = torch.cat(x_grad_seq_list, dim=1)

    # Backward result check
    if use_flash_attn:
        assert_close(qkv_grad_sp, qkv_grad_no_sp, atol=5e-2, rtol=1e-2)
        assert_close(o_grad_sp, o_grad_no_sp, atol=5e-3, rtol=5e-3)
        assert_close(x_grad_seq_gather, x_unshard_grad, atol=1e-3, rtol=1e-3)
    else:
        # assert_close(qkv_grad_sp, qkv_grad_no_sp, atol=5e-5, rtol=5e-5)
        assert_close(o_grad_sp, o_grad_no_sp, atol=5e-5, rtol=5e-5)
        assert_close(x_grad_seq_gather, x_unshard_grad, atol=5e-5, rtol=5e-5)


@parameterize("seq_len", [256])
@parameterize("hidden_dim", [1152])
@parameterize("head_num", [16])
@parameterize("batch_size", [2])
@parameterize("sequence_parallel_type", ["longseq"])
@parameterize("sequence_parallel_overlap", [False])
@parameterize("use_flash_attn", [False])
def run_seq_parallel_attn(
    seq_len, hidden_dim, head_num, batch_size, sequence_parallel_type, sequence_parallel_overlap, use_flash_attn
):
    seq_parallel_attn(
        seq_len, hidden_dim, head_num, batch_size, sequence_parallel_type, sequence_parallel_overlap, use_flash_attn
    )


def check_longseq_attn(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_seq_parallel_attn()


@rerun_if_address_is_in_use()
def test_sequence_parallel_attn():
    spawn(check_longseq_attn, nprocs=WORKERS)


if __name__ == "__main__":
    test_sequence_parallel_attn()
