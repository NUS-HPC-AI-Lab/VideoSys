import torch
import triton
from torch.cuda.amp import custom_bwd, custom_fwd

from .k_fused_modulate import fused_modulate_bwd, fused_modulate_fwd


class _FusedModulate(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, x, scale, shift):
        shift = shift.contiguous()
        scale = scale.contiguous()

        # allocate output
        y = torch.empty_like(x)

        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape

        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE_N:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

        if not x_arg.is_contiguous() or not y.is_contiguous():
            x_arg = x_arg.contiguous()
            y = y.contiguous()

        # heuristics for number of warps.
        num_warps = min(max(BLOCK_SIZE_N // 256, 1), 16)

        # enqueue kernel
        # fmt: off
        fused_modulate_fwd[(M,)](
            x_arg, y,
            scale, shift, y.stride(1),
            x_arg.stride(0),
            N,
            num_warps=num_warps,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
        )
        # fmt: on

        ctx.save_for_backward(x, scale)
        ctx.BLOCK_SIZE_N = BLOCK_SIZE_N
        ctx.num_warps = num_warps

        return y.reshape_as(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, dy):  # pragma: no cover  # this is covered, but called directly from C++
        x, scale = ctx.saved_tensors

        # flatten the batch dimension, if any.
        # We're interested in 'samples' x norm_dimension
        x = x.reshape(-1, x.size(-1))
        M, N = x.size()

        # allocate output
        dy = dy.contiguous()
        dx = torch.empty_like(dy)
        d_scale = torch.empty_like(dy)
        d_shift = torch.sum(dy, dim=-2)

        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW and DB
        num_warps = min(max(ctx.BLOCK_SIZE_N // 256, 1), 16)

        # fmt: off
        fused_modulate_bwd[(M,)](
            dx, dy, d_scale, scale, x,
            x.stride(0) , dy.stride(1) ,
            N,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE_N,
            num_warps=num_warps
        )
        # fmt: on

        dx = dx.reshape_as(dy)
        d_scale = torch.sum(d_scale, dim=-2)
        return dx, d_scale, d_shift


def fused_modulate(
    x: torch.Tensor,
    scale=None,
    shift=None,
) -> torch.Tensor:
    return _FusedModulate.apply(x, scale, shift)
