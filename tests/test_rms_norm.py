import torch
from apex.normalization.fused_layer_norm import FusedRMSNorm

from videosys.models.modules.normalization import LlamaRMSNorm

device = torch.device(0)
dtype = torch.float
shape = [4, 4096, 16, 128]

act1 = torch.rand(shape, device=device, dtype=dtype, requires_grad=True)
act2 = act1.detach().clone().requires_grad_(True)
act3 = act1.detach().clone().requires_grad_(True)

grad = torch.rand_like(act1)

ref_norm = LlamaRMSNorm(shape[-1]).to(device, dtype)
# torch_norm = RMSNorm(shape[-1]).to(device, dtype)
apex_norm = FusedRMSNorm(shape[-1]).to(device, dtype)

out1 = ref_norm(act1)
out1.backward(grad)

# out2 = torch_norm(act2)
# out2.backward(grad)

out3 = apex_norm(act3)
out3.backward(grad)

# if not torch.allclose(out1, out2, rtol=1e-3, atol=1e-4):
#     diff = torch.abs(out1 - out2).norm()
#     print(f"torch RMSNorm forward failed, norm: {diff.item()}")

# if not torch.allclose(act1.grad, act2.grad, rtol=1e-3, atol=1e-4):
#     diff = torch.abs(act1.grad - act2.grad).norm()
#     print(f"torch RMSNorm backward failed, norm: {diff.item()}")

if not torch.allclose(out1, out3, rtol=1e-3, atol=1e-4):
    diff = torch.abs(out1 - out3)
    print(f"apex RMSNorm failed, norm: {diff.norm().item()}, max: {diff.max().item()}, min: {diff.min().item()}")

if not torch.allclose(act1.grad, act3.grad, rtol=1e-3, atol=1e-4):
    diff = torch.abs(act1.grad - act3.grad)
    print(f"apex RMSNorm failed, norm: {diff.norm().item()}, max: {diff.max().item()}, min: {diff.min().item()}")
