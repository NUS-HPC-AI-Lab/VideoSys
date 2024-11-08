import argparse

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func


def flash_attn(q, k, v, dropout_p=0.0):
    return flash_attn_func(q, k, v, dropout_p)


def torch_attn(q, k, v):
    head_dim = q.size(-1)
    dtype = q.dtype
    q = q * head_dim**-0.5
    attn = q @ k.transpose(-2, -1)  # translate attn to float32
    attn = attn.to(torch.float32)
    attn = attn.softmax(dim=-1)
    attn = attn.to(dtype)  # cast back attn to original dtype
    o = attn @ v
    return o


def main(args):
    bs = args.batch_size
    num_head = args.num_head
    seq_len = args.seq_len
    head_dim = args.head_dim
    device = torch.device("cuda:0")
    dtype = torch.bfloat16

    if args.vendor == 'torch':
        shape = (bs, num_head, seq_len, head_dim)
        attn_func = torch_attn
    else:
        shape = (bs, seq_len, num_head, head_dim)
        attn_func = flash_attn
    
    q = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(shape, device=device, dtype=dtype, requires_grad=True)
    grad = torch.randn(shape, device=device, dtype=dtype)

    print(f"{args.vendor} init: {torch.cuda.memory_allocated()/1024**2}, {torch.cuda.max_memory_allocated()/1024**2}")
    o = attn_func(q, k, v)
    print(f"{args.vendor} forward: {torch.cuda.memory_allocated()/1024**2}, {torch.cuda.max_memory_allocated()/1024**2}")
    o.backward(grad)
    torch.cuda.synchronize()
    print(f"{args.vendor} backward: {torch.cuda.memory_allocated()/1024**2}, {torch.cuda.max_memory_allocated()/1024**2}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_head', type=int)
    parser.add_argument('--seq_len', type=int)
    parser.add_argument('--head_dim', type=int)
    parser.add_argument('--vendor', type=str, choices=['torch', 'flash'])
    args = parser.parse_args()
    
    main(args)
