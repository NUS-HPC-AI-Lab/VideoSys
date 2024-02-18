import torch
import torch.distributed as dist


# using all_to_all_single api to perform all to all communication
def _all_to_all_single(input_, seq_world_size, group, scatter_dim, gather_dim):
    inp_shape = list(input_.shape)
    inp_shape[scatter_dim] = inp_shape[scatter_dim] // seq_world_size
    if scatter_dim < 2:
        input_t = input_.reshape([seq_world_size, inp_shape[scatter_dim]] + inp_shape[scatter_dim + 1 :]).contiguous()
    else:
        input_t = (
            input_.reshape([-1, seq_world_size, inp_shape[scatter_dim]] + inp_shape[scatter_dim + 1 :])
            .transpose(0, 1)
            .contiguous()
        )

    output = torch.empty_like(input_t)
    dist.all_to_all_single(output, input_t, group=group)

    if scatter_dim < 2:
        output = output.transpose(0, 1).contiguous()

    return output.reshape(
        inp_shape[:gather_dim]
        + [
            inp_shape[gather_dim] * seq_world_size,
        ]
        + inp_shape[gather_dim + 1 :]
    ).contiguous()


# using all_to_all api to perform all to all communication
def _all_to_all(input_, world_size, group, scatter_dim, gather_dim):
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
        bsz, _, _ = input_.shape

        # Todo: Try to make all_to_all_single compatible with a large batch size
        if bsz == 1:
            return _all_to_all_single(input_, world_size, process_group, scatter_dim, gather_dim)
        else:
            return _all_to_all(input_, world_size, process_group, scatter_dim, gather_dim)

    @staticmethod
    def backward(ctx, *grad_output):
        process_group = ctx.process_group
        scatter_dim = ctx.gather_dim
        gather_dim = ctx.scatter_dim
        return_grad = _AllToAll.apply(*grad_output, process_group, scatter_dim, gather_dim)
        return (return_grad, None, None, None)
