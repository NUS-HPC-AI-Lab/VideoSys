import torch

from videosys.kernels.fused_modulate import fused_modulate


def test_fused_modulate():
    x1 = torch.rand((1, 20, 100), requires_grad=True).cuda()
    x1.retain_grad()
    shift1 = torch.rand((1, 100), requires_grad=True).cuda()
    shift1.retain_grad()
    scale1 = torch.rand((1, 100), requires_grad=True).cuda()
    scale1.retain_grad()
    x2 = x1.clone().detach().requires_grad_()
    shift2 = shift1.clone().detach().requires_grad_()
    scale2 = scale1.clone().detach().requires_grad_()

    out1 = fused_modulate(x1, scale1, shift1)
    out1.mean().backward()
    out2 = x2 * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
    out2.mean().backward()

    assert torch.allclose(out1, out2, atol=1e-6), f"\nout1:\n{out1}\nout2:\n{out2}\n"
    assert torch.allclose(x1.grad, x2.grad, atol=1e-6), f"\nx1.grad:\n{x1.grad}\nx2.grad:\n{x2.grad}\n"
    assert torch.allclose(
        scale1.grad, scale2.grad, atol=1e-4
    ), f"\nscale1.grad:\n{scale1.grad}\nscale2.grad:\n{scale2.grad}\n"
    assert torch.allclose(
        shift1.grad, shift2.grad, atol=1e-4
    ), f"\nshift1.grad:\n{shift1.grad}\nshift2.grad:\n{shift2.grad}\n"


if __name__ == "__main__":
    test_fused_modulate()
