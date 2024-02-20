import triton
import triton.language as tl


@triton.jit
def fused_modulate_fwd(X, Y, SCALE, SHIFT, seq_stride, stride, N, BLOCK_SIZE_N: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N

    # get scale and shift
    scale_ptrs = SCALE + row // seq_stride * stride + cols
    scale = tl.load(scale_ptrs, mask=mask, other=0.0)
    shift_ptrs = SHIFT + row // seq_stride * stride + cols
    shift = tl.load(shift_ptrs, mask=mask, other=0.0)

    # Move to this row
    x_ptrs = X + row * stride + cols
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    # scale and shift
    y = x * (1 + scale) + shift

    mask = cols < N
    y_ptrs = Y + row * stride + cols
    tl.store(y_ptrs, y, mask=mask)


@triton.jit
def fused_modulate_bwd(
    DX,
    DY,
    D_SCALE,
    SCALE,
    X,
    stride,
    seq_stride,
    N,
    BLOCK_SIZE_N: tl.constexpr,
):
    # position of elements processed by this program
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N

    # offset data pointers to start at the row of interest
    x_ptrs = X + row * stride + cols
    dy_ptrs = DY + row * stride + cols

    # scale and shift
    scale_ptrs = SCALE + (row // seq_stride) * stride + cols
    scale = tl.load(scale_ptrs, mask=mask, other=0.0)
    dscale_ptrs = D_SCALE + row * stride + cols

    # load data to SRAM
    x = tl.load(x_ptrs, mask=mask, other=0)
    dy = tl.load(dy_ptrs, mask=mask, other=0)

    # unscale
    dx = dy * (1 + scale)
    dscale = dy * x
    tl.store(dscale_ptrs, dscale, mask=mask)

    # write-back dx
    cols = tl.arange(0, BLOCK_SIZE_N)
    mask = cols < N  # re-materialize the mask to save registers
    dx_ptrs = DX + row * stride + cols
    tl.store(dx_ptrs, dx, mask=mask)
