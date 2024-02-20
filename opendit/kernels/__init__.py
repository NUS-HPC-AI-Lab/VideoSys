try:
    import triton  # noqa
    import triton.language as tl  # noqa

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    from .fused_modulate import fused_modulate

    __all__ = ["fused_modulate"]
else:
    __all__ = []
