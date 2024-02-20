import torch
import triton

from .k_fused_modulate import _modulate_bwd, _modulate_fwd


class _FusedModulate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale, shift):
        # allocate output
        y = torch.empty_like(x)
        n_elements = y.numel()
        s_stride = y.shape[1]
        x = x.contiguous()
        scale = scale.unsqueeze(1).repeat(1, s_stride, 1).contiguous()
        shift = shift.unsqueeze(1).repeat(1, s_stride, 1).contiguous()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _modulate_fwd[grid](x, y, scale, shift, n_elements, BLOCK_SIZE=1024)

        ctx.save_for_backward(x)
        return y

    @staticmethod
    def backward(ctx, dy):  # pragma: no cover  # this is covered, but called directly from C++
        (x,) = ctx.saved_tensors

        # allocate output
        dy = dy.contiguous()
        dx = torch.empty_like(dy)

        n_elements = x.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        _modulate_bwd[grid](dx, dy, n_elements, BLOCK_SIZE=1024)

        return dx, dx, dx


def fused_modulate(
    x: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
) -> torch.Tensor:
    return _FusedModulate.apply(x, scale, shift)
