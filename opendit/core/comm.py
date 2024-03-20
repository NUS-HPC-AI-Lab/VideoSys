from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.distributed import ProcessGroup

# ======================================================
# Model
# ======================================================


def model_sharding(model: torch.nn.Module):
    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    for _, param in model.named_parameters():
        padding_size = (world_size - param.numel() % world_size) % world_size
        if padding_size > 0:
            padding_param = torch.nn.functional.pad(param.data.view(-1), [0, padding_size])
        else:
            padding_param = param.data.view(-1)
        splited_params = padding_param.split(padding_param.numel() // world_size)
        splited_params = splited_params[global_rank]
        param.data = splited_params


# ======================================================
# AllGather & ReduceScatter
# ======================================================


class AsyncAllGatherForTwo(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        weight: Tensor,
        bias: Tensor,
        sp_rank: int,
        sp_size: int,
        group: Optional[ProcessGroup] = None,
    ) -> Tuple[Tensor, Any]:
        """
        Returns:
            outputs: Tensor
            handle: Optional[Work], if overlap is True
        """
        from torch.distributed._functional_collectives import all_gather_tensor

        ctx.group = group
        ctx.sp_rank = sp_rank
        ctx.sp_size = sp_size

        # all gather inputs
        all_inputs = all_gather_tensor(inputs.unsqueeze(0), 0, group)
        # compute local qkv
        local_qkv = F.linear(inputs, weight, bias).unsqueeze(0)

        # remote compute
        remote_inputs = all_inputs[1 - sp_rank].view(list(local_qkv.shape[:-1]) + [-1])
        # compute remote qkv
        remote_qkv = F.linear(remote_inputs, weight, bias)

        # concat local and remote qkv
        if sp_rank == 0:
            qkv = torch.cat([local_qkv, remote_qkv], dim=0)
        else:
            qkv = torch.cat([remote_qkv, local_qkv], dim=0)
        qkv = rearrange(qkv, "sp b n c -> b (sp n) c")

        ctx.save_for_backward(inputs, weight, remote_inputs)
        return qkv

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Tuple[Tensor, None, None]:
        from torch.distributed._functional_collectives import reduce_scatter_tensor

        group = ctx.group
        sp_rank = ctx.sp_rank
        sp_size = ctx.sp_size
        inputs, weight, remote_inputs = ctx.saved_tensors

        # split qkv_grad
        qkv_grad = grad_outputs[0]
        qkv_grad = rearrange(qkv_grad, "b (sp n) c -> sp b n c", sp=sp_size)
        qkv_grad = torch.chunk(qkv_grad, 2, dim=0)
        if sp_rank == 0:
            local_qkv_grad, remote_qkv_grad = qkv_grad
        else:
            remote_qkv_grad, local_qkv_grad = qkv_grad

        # compute remote grad
        remote_inputs_grad = torch.matmul(remote_qkv_grad, weight).squeeze(0)
        weight_grad = torch.matmul(remote_qkv_grad.transpose(-1, -2), remote_inputs).squeeze(0).sum(0)
        bias_grad = remote_qkv_grad.squeeze(0).sum(0).sum(0)

        # launch async reduce scatter
        remote_inputs_grad_zero = torch.zeros_like(remote_inputs_grad)
        if sp_rank == 0:
            remote_inputs_grad = torch.cat([remote_inputs_grad_zero, remote_inputs_grad], dim=0)
        else:
            remote_inputs_grad = torch.cat([remote_inputs_grad, remote_inputs_grad_zero], dim=0)
        remote_inputs_grad = reduce_scatter_tensor(remote_inputs_grad, "sum", 0, group)

        # compute local grad and wait for reduce scatter
        local_input_grad = torch.matmul(local_qkv_grad, weight).squeeze(0)
        weight_grad += torch.matmul(local_qkv_grad.transpose(-1, -2), inputs).squeeze(0).sum(0)
        bias_grad += local_qkv_grad.squeeze(0).sum(0).sum(0)

        # sum remote and local grad
        inputs_grad = remote_inputs_grad + local_input_grad
        return inputs_grad, weight_grad, bias_grad, None, None, None


class AllGather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        group: Optional[ProcessGroup] = None,
        overlap: bool = False,
    ) -> Tuple[Tensor, Any]:
        """
        Returns:
            outputs: Tensor
            handle: Optional[Work], if overlap is True
        """
        assert ctx is not None or not overlap

        if ctx is not None:
            ctx.comm_grp = group

        comm_size = dist.get_world_size(group)
        if comm_size == 1:
            return inputs.unsqueeze(0), None

        buffer_shape = (comm_size,) + inputs.shape
        outputs = torch.empty(buffer_shape, dtype=inputs.dtype, device=inputs.device)
        buffer_list = list(torch.chunk(outputs, comm_size, dim=0))
        if not overlap:
            dist.all_gather(buffer_list, inputs, group=group)
            return outputs, None
        else:
            handle = dist.all_gather(buffer_list, inputs, group=group, async_op=True)
            return outputs, handle

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Tuple[Tensor, None, None]:
        return (
            ReduceScatter.forward(None, grad_outputs[0], ctx.comm_grp, False)[0],
            None,
            None,
        )


class ReduceScatter(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        inputs: Tensor,
        group: ProcessGroup,
        overlap: bool = False,
    ) -> Tuple[Tensor, Any]:
        """
        Returns:
            outputs: Tensor
            handle: Optional[Work], if overlap is True
        """
        assert ctx is not None or not overlap

        if ctx is not None:
            ctx.comm_grp = group

        comm_size = dist.get_world_size(group)
        if comm_size == 1:
            return inputs.squeeze(0), None

        if not inputs.is_contiguous():
            inputs = inputs.contiguous()

        output_shape = inputs.shape[1:]
        outputs = torch.empty(output_shape, dtype=inputs.dtype, device=inputs.device)
        buffer_list = list(torch.chunk(inputs, comm_size, dim=0))
        if not overlap:
            dist.reduce_scatter(outputs, buffer_list, group=group)
            return outputs, None
        else:
            handle = dist.reduce_scatter(outputs, buffer_list, group=group, async_op=True)
            return outputs, handle

    @staticmethod
    def backward(ctx: Any, *grad_outputs) -> Tuple[Tensor, None, None]:
        # TODO: support async backward
        return (
            AllGather.forward(None, grad_outputs[0], ctx.comm_grp, False)[0],
            None,
            None,
        )


# ======================================================
# AlltoAll
# ======================================================


def _all_to_all_func(input_, world_size, group, scatter_dim, gather_dim):
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        world_size = dist.get_world_size(process_group)

        return _all_to_all_func(input_, world_size, process_group, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx, *grad_output):
        process_group = ctx.process_group
        scatter_dim = ctx.gather_dim
        gather_dim = ctx.scatter_dim
        return_grad = _AllToAll.apply(*grad_output, process_group, scatter_dim, gather_dim)
        return (return_grad, None, None, None)


def all_to_all_comm(input_, process_group=None, scatter_dim=2, gather_dim=1):
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim)


# ======================================================
# Sequence Gather & Split
# ======================================================


def _split_sequence_func(inputs, pg: dist.ProcessGroup, dim=-1):
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return inputs

    # Split along last dimension.
    rank = dist.get_rank(pg)
    dim_size = inputs.size(dim)
    assert dim_size % world_size == 0, (
        f"The dimension to split ({dim_size}) is not a multiple of world size ({world_size}), "
        f"cannot split tensor evenly"
    )

    outputs = torch.split(inputs, dim_size // world_size, dim=dim)[rank]
    return outputs


def _gather_sequence_func(inputs, pg: dist.ProcessGroup, dim=-1):
    world_size = dist.get_world_size(pg)
    if world_size == 1:
        return inputs

    # all gather
    inputs = inputs.contiguous()
    outputs = [torch.empty_like(inputs) for _ in range(world_size)]
    dist.all_gather(outputs, inputs, group=pg)

    # concat
    outputs = torch.cat(outputs, dim=dim)
    return outputs


class _GatherForwardSplitBackward(torch.autograd.Function):
    """
    Gather the input sequence.

    Args:
        input_: input matrix.
        process_group: process group.
        dim: dimension
    """

    @staticmethod
    def symbolic(graph, input_):
        return _gather_sequence_func(input_)

    @staticmethod
    def forward(ctx, input_, process_group, dim, grad_scale):
        ctx.process_group = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        return _gather_sequence_func(input_, process_group, dim)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.process_group)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.process_group)

        return _split_sequence_func(grad_output, ctx.process_group, ctx.dim), None, None, None


class _SplitForwardGatherBackward(torch.autograd.Function):
    """
    Split sequence.

    Args:
        input_: input matrix.
        process_group: parallel mode.
        dim: dimension
    """

    @staticmethod
    def symbolic(graph, input_):
        return _split_sequence_func(input_)

    @staticmethod
    def forward(ctx, input_, process_group, dim, grad_scale):
        ctx.process_group = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        return _split_sequence_func(input_, process_group, dim)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.process_group)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.process_group)
        return _gather_sequence_func(grad_output, ctx.process_group, ctx.dim), None, None, None


def split_sequence(input_, process_group, dim, grad_scale=1.0):
    return _SplitForwardGatherBackward.apply(input_, process_group, dim, grad_scale)


def gather_sequence(input_, process_group, dim, grad_scale=None):
    return _GatherForwardSplitBackward.apply(input_, process_group, dim, grad_scale)
