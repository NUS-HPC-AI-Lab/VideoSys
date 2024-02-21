import math

import torch
from colossalai.kernel.extensions.flash_attention import HAS_FLASH_ATTN, HAS_MEM_EFF_ATTN
from colossalai.testing import clear_cache_before_run, parameterize
from einops import rearrange

print("HAS_FLASH_ATTN", HAS_FLASH_ATTN)

if HAS_MEM_EFF_ATTN or HAS_FLASH_ATTN:
    from colossalai.nn.layer.colo_attention import ColoAttention

DTYPE = [torch.float16, torch.bfloat16]


def attention_ref(q, k, v, attn_mask=None, causal=False):
    """
    attention output of the control group
    """
    dtype_og = q.dtype
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    scores = torch.einsum("bthd,bshd->bhts", q * scale, k)

    if attn_mask is not None:
        scores.masked_fill_(rearrange(~attn_mask, "b s -> b 1 1 s"), float("-inf"))
    if causal:
        causal_mask = torch.triu(torch.ones(seqlen_q, seqlen_k, dtype=torch.bool, device=q.device), 1)
        scores.masked_fill_(causal_mask, float("-inf"))
    attention = torch.softmax(scores, dim=-1)

    output = torch.einsum("bhts,bshd->bthd", attention, v)
    output = rearrange(output, "b s h d -> b s (h d)")

    # Modify the data at the positions of the mask to 0
    if attn_mask is not None:
        output.masked_fill_(rearrange(~attn_mask, "b s -> b s 1"), 0.0)

    return output.to(dtype=dtype_og)


# @pytest.mark.skipif(not HAS_MEM_EFF_ATTN and not HAS_FLASH_ATTN, reason="xformers is not available")
@clear_cache_before_run()
@parameterize("proj_shape", [(6, 8, 4, 16)])
@parameterize("dtype", DTYPE)
@parameterize("dropout", [0.0])
def test_attention_no_mask(proj_shape, dtype, dropout):
    (B, S, H, D_HEAD) = proj_shape
    D = H * D_HEAD

    q = torch.randn((B, S, H, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
    k = torch.randn((B, S, H, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)
    v = torch.randn((B, S, H, D_HEAD), dtype=dtype, device="cuda", requires_grad=True)

    attn = ColoAttention(D, H, dropout=dropout)
    y = attn(q, k, v)

    assert list(y.shape) == [B, S, D]

    out_ref = attention_ref(q, k, v, None, causal=False)

    dy = torch.rand_like(y)
    grad_q, grad_k, grad_v = torch.autograd.grad(y, (q, k, v), dy)
    grad_ref_q, grad_ref_k, grad_ref_v = torch.autograd.grad(out_ref, (q, k, v), dy)

    torch.allclose(y, out_ref, atol=1e-7), f"{(y - out_ref).abs().max()}"
    torch.allclose(grad_q, grad_ref_q, atol=1e-7), f"{(grad_q - grad_ref_q).abs().max()}"
    torch.allclose(grad_k, grad_ref_k, atol=1e-7), f"{(grad_k - grad_ref_k).abs().max()}"
    torch.allclose(grad_v, grad_ref_v, atol=1e-7), f"{(grad_v - grad_ref_v).abs().max()}"
