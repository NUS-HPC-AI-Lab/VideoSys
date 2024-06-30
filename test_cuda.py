import torch
import torch.version

print(torch.__version__)

print(torch.cuda.is_available())

print(torch.cuda.device_count())

print(torch.version.cuda)
