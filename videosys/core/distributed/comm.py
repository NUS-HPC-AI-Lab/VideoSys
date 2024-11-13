from typing import Any, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

# ======================================================
# AllGather & ReduceScatter
# ======================================================


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


def _split_sequence_func(input_, pg: dist.ProcessGroup, dim: int, pad: int, pad_val: int = 0):
    # skip if only one rank involved
    world_size = dist.get_world_size(pg)
    rank = dist.get_rank(pg)
    if world_size == 1:
        return input_

    if pad > 0:
        pad_size = list(input_.shape)
        pad_size[dim] = pad
        input_ = torch.cat(
            [input_, torch.empty(pad_size, dtype=input_.dtype, device=input_.device).fill_(pad_val)], dim=dim
        )

    dim_size = input_.size(dim)
    assert dim_size % world_size == 0, f"dim_size ({dim_size}) is not divisible by world_size ({world_size})"

    tensor_list = torch.split(input_, dim_size // world_size, dim=dim)
    output = tensor_list[rank].contiguous()
    return output


def _gather_sequence_func(input_, pg: dist.ProcessGroup, dim: int, pad: int):
    # skip if only one rank involved
    input_ = input_.contiguous()
    world_size = dist.get_world_size(pg)
    dist.get_rank(pg)

    if world_size == 1:
        return input_

    # all gather
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    assert input_.device.type == "cuda"
    torch.distributed.all_gather(tensor_list, input_, group=pg)

    # concat
    output = torch.cat(tensor_list, dim=dim)

    if pad > 0:
        output = output.narrow(dim, 0, output.size(dim) - pad)

    return output


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
    def forward(ctx, input_, process_group, dim, grad_scale, pad):
        ctx.process_group = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        ctx.pad = pad
        return _gather_sequence_func(input_, process_group, dim, pad)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.process_group)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.process_group)

        return _split_sequence_func(grad_output, ctx.process_group, ctx.dim, ctx.pad), None, None, None, None


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
    def forward(ctx, input_, process_group, dim, grad_scale, pad, pad_val):
        ctx.process_group = process_group
        ctx.dim = dim
        ctx.grad_scale = grad_scale
        ctx.pad = pad
        return _split_sequence_func(input_, process_group, dim, pad, pad_val)

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.grad_scale == "up":
            grad_output = grad_output * dist.get_world_size(ctx.process_group)
        elif ctx.grad_scale == "down":
            grad_output = grad_output / dist.get_world_size(ctx.process_group)
        return _gather_sequence_func(grad_output, ctx.process_group, ctx.dim, ctx.pad), None, None, None, None, None


def split_sequence(input_, process_group, dim, grad_scale=1.0, pad=0, pad_val=0):
    return _SplitForwardGatherBackward.apply(input_, process_group, dim, grad_scale, pad, pad_val)


def gather_sequence(input_, process_group, dim, grad_scale=1.0, pad=0):
    return _GatherForwardSplitBackward.apply(input_, process_group, dim, grad_scale, pad)


# ==============================
# Pad
# ==============================

PAD_DICT = {}


def set_pad(name: str, dim_size: int, parallel_group: dist.ProcessGroup):
    sp_size = dist.get_world_size(parallel_group)
    pad = (sp_size - (dim_size % sp_size)) % sp_size
    global PAD_DICT
    PAD_DICT[name] = pad


def get_pad(name) -> int:
    return PAD_DICT[name]


def all_to_all_with_pad(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
    scatter_pad: int = 0,
    gather_pad: int = 0,
):
    if scatter_pad > 0:
        pad_shape = list(input_.shape)
        pad_shape[scatter_dim] = scatter_pad
        pad_tensor = torch.zeros(pad_shape, device=input_.device, dtype=input_.dtype)
        input_ = torch.cat([input_, pad_tensor], dim=scatter_dim)

    assert (
        input_.shape[scatter_dim] % dist.get_world_size(process_group) == 0
    ), f"Dimension to scatter ({input_.shape[scatter_dim]}) is not divisible by world size ({dist.get_world_size(process_group)})"
    input_ = _AllToAll.apply(input_, process_group, scatter_dim, gather_dim)

    if gather_pad > 0:
        input_ = input_.narrow(gather_dim, 0, input_.size(gather_dim) - gather_pad)

    return input_


def split_from_second_dim(x, batch_size, parallel_group):
    x = x.view(batch_size, -1, *x.shape[1:])
    x = split_sequence(x, parallel_group, dim=1, grad_scale="down", pad=get_pad("temporal"))
    x = x.reshape(-1, *x.shape[2:])
    return x


def gather_from_second_dim(x, batch_size, parallel_group):
    x = x.view(batch_size, -1, *x.shape[1:])
    x = gather_sequence(x, parallel_group, dim=1, grad_scale="up", pad=get_pad("temporal"))
    x = x.reshape(-1, *x.shape[2:])
    return x
