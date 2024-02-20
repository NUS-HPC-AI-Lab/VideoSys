# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.
#
# This source code is licensed under the BSD license found in the
# LICENSE file in the root directory of this source tree.

# CREDITS: the underlying kernel comes straight from the Triton tutorials
# see https://github.com/openai/triton/blob/master/python/tutorials/05-layer-norm.py

import logging
from typing import Optional

import torch
import torch.nn as nn
import triton
import triton.language as tl
from k_layernorm import layer_norm_bwd_dwdb, layer_norm_bwd_dx_fused, layer_norm_fw
from torch.cuda.amp import custom_bwd, custom_fwd

logger = logging.getLogger("xformers")


_triton_layernorm_fp16_enabled = False  # NOTE: PyTorch keeps layernorm as fp32
_triton_registered_warnings = False


class _LayerNorm(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float16 if _triton_layernorm_fp16_enabled else None)
    def forward(ctx, x, weight, bias, eps, scale, shift):
        # catch eps being too small if the tensors are fp16
        if x.dtype == torch.float16:
            eps = max(eps, 1.6e-5)

        # allocate output
        y = torch.empty_like(x)

        # reshape input data into 2D tensor
        x_arg = x.reshape(-1, x.shape[-1])
        M, N = x_arg.shape

        # allocate mean and std, they'll be used in the backward pass
        mean = torch.empty((M,), dtype=torch.float32, device="cuda")
        rstd = torch.empty((M,), dtype=torch.float32, device="cuda")

        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_SIZE_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(N))
        if N > BLOCK_SIZE_N:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")

        if not x_arg.is_contiguous() or not y.is_contiguous():
            global _triton_registered_warnings
            if not _triton_registered_warnings:
                logger.warning(
                    "Non-contiguous input tensor found. Making it contiguous,"
                    + " but could have perf or trainer implications"
                )

                _triton_registered_warnings = True

            x_arg = x_arg.contiguous()
            y = y.contiguous()

        # heuristics for number of warps.
        num_warps = min(max(BLOCK_SIZE_N // 256, 1), 16)

        # enqueue kernel
        # fmt: off
        layer_norm_fw[(M,)](
            x_arg, y, weight, bias, mean, rstd,
            scale, shift, y.stride(1),
            x_arg.stride(0),
            N,
            eps,
            num_warps=num_warps,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            affine=weight is not None
        )
        # fmt: on

        ctx.save_for_backward(x, mean, rstd, weight, scale, shift)
        ctx.BLOCK_SIZE_N = BLOCK_SIZE_N
        ctx.num_warps = num_warps

        return y.reshape_as(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, dy):  # pragma: no cover  # this is covered, but called directly from C++
        x, mean, rstd, weight, scale, shift = ctx.saved_tensors

        # flatten the batch dimension, if any.
        # We're interested in 'samples' x norm_dimension
        x = x.reshape(-1, x.size(-1))
        M, N = x.size()

        # heuristics for amount of parallel reduction stream for DG/DB
        GROUP_SIZE_M = 32
        if N <= 8192:
            GROUP_SIZE_M = 64
        if N <= 4096:
            GROUP_SIZE_M = 96
        if N <= 2048:
            GROUP_SIZE_M = 128
        if N <= 1024:
            GROUP_SIZE_M = 256

        if dy.dtype == torch.float32:
            GROUP_SIZE_M = GROUP_SIZE_M // 2

        # allocate output
        locks = torch.zeros(2 * GROUP_SIZE_M, dtype=torch.int32, device="cuda")
        t_args = {"dtype": x.dtype, "device": x.device}
        _dw = torch.empty((GROUP_SIZE_M, x.size(-1)), **t_args)
        _db = torch.empty_like(_dw)
        dw = torch.empty((x.size(-1),), **t_args)
        db = torch.empty_like(dw)
        dy = dy.contiguous()
        dx = torch.empty_like(dy)
        d_scale = torch.empty_like(dy)
        d_shift = torch.sum(dy, dim=-2)

        # Check the tensor shapes and layouts
        # we suppose in the kernel that they have the same size and are contiguous
        assert (
            dy.numel() == x.numel()
        ), "Something is wrong in the backward graph, possibly because of an inplace operation after the layernorm"

        # enqueue kernel using forward pass heuristics
        # also compute partial sums for DW and DB
        num_warps = min(max(ctx.BLOCK_SIZE_N // 256, 1), 16)

        # fmt: off
        layer_norm_bwd_dx_fused[(M,)](
            dx, dy, _dw, _db, d_scale, scale, x,
            weight if weight is not None else x,
            mean, rstd,
            locks,
            x.stride(0), dy.stride(1),
            N,
            affine=weight is not None,
            GROUP_SIZE_M=GROUP_SIZE_M,
            BLOCK_SIZE_N=ctx.BLOCK_SIZE_N,
            num_warps=num_warps
        )
        # fmt: on

        def grid(meta):
            return [triton.cdiv(N, meta["BLOCK_SIZE_N"])]

        # accumulate partial sums in separate kernel
        # fmt: off
        layer_norm_bwd_dwdb[grid](
            _dw, _db, dw, db,
            GROUP_SIZE_M,
            N,
            BLOCK_SIZE_M=32,
            BLOCK_SIZE_N=64
        )
        # fmt: on

        dx = dx.reshape_as(dy)
        d_scale = torch.sum(d_scale, dim=-2)
        return dx, dw, db, None, d_scale, d_shift


class FusedLayerNorm(nn.Module):
    """
    Handle a layer normalization, like torch.nn.LayerNorm_.

    This implementation should be measurably faster than the default PyTorch layernorm (as of PyTorch 1.9),
    both for training and inference worloads.

    .. NOTE: Computations under Torch AMP are kept as float32 by default, one can change this to be float16
        by setting the flag `xformers.triton.k_layer_norm._triton_layernorm_fp16_enabled = True`

    .. _torch.nn.LayerNorm: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html

    """

    def __init__(self, normalized_shape, affine=True, eps=1e-06):
        super().__init__()
        if affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.weight = self.bias = None
        self.epsilon = eps

    def forward(self, x):
        return layer_norm(x, self.weight, self.bias, self.epsilon)

    def init_weights(self, *args, **kwargs):
        with torch.no_grad():
            if self.weight is not None:
                self.weight.fill_(1.0)

            if self.bias is not None:
                self.bias.fill_(0.0)


def layer_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-06,
    scale=None,
    shift=None,
) -> torch.Tensor:
    return _LayerNorm.apply(x, weight, bias, eps, scale, shift)


# Backward pass (total DW + total DB)
# fmt: off
@triton.jit
def layer_norm_bwd_dwdb(
    DW, DB, FINAL_DW, FINAL_DB,
    M, N,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr
):
    # fmt: on

    pid = tl.program_id(0)

    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_cols = cols < N

    dw = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    db = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for i in range(0, M, BLOCK_SIZE_M):
        rows = i + tl.arange(0, BLOCK_SIZE_M)
        offs = rows[:, None] * N + cols[None, :]
        mask_rm = rows < M

        dw += tl.load(DW + offs, mask=mask_rm[:, None] & mask_cols[None, :], other=0.0)
        db += tl.load(DB + offs, mask=mask_rm[:, None] & mask_cols[None, :], other=0.0)

    sum_dw = tl.sum(dw, axis=0)
    sum_db = tl.sum(db, axis=0)

    cols = pid * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    mask_cols = cols < N

    tl.store(FINAL_DW + cols, sum_dw, mask=mask_cols)
    tl.store(FINAL_DB + cols, sum_db, mask=mask_cols)


if __name__ == "__main__":
    x = torch.rand((1, 100, 1024), requires_grad=True).cuda()
    shift = torch.rand((1, 1024), requires_grad=True).cuda()
    scale = torch.rand((1, 1024), requires_grad=True).cuda()
    a = layer_norm(x.clone(), None, None, 1e-5, scale.clone(), shift.clone())
    a.mean().backward()
    c = torch.nn.functional.layer_norm(x.clone(), (1024,), weight=None, bias=None, eps=1e-5) * (1 + scale.clone().unsqueeze(1)) + shift.clone().unsqueeze(1)
    c.mean().backward()
    # cc = triton.testing.do_bench(lambda: torch.nn.functional.layer_norm(x, (1024,), weight=w, bias=b, eps=1e-5) * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1))
    print(a - c)
