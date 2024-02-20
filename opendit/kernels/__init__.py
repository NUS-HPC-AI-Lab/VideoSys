try:
    import triton  # noqa
    import triton.language as tl  # noqa

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    from .layernorm import FusedLayerNorm

    __all__ = ["FusedLayerNorm"]
else:
    __all__ = []
