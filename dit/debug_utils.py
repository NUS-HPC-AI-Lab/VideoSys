import torch.distributed as dist

def print_rank(var_name, var_value, rank=0):
    if dist.get_rank() == 0:
        print(f'[Rank {rank}] {var_name}: {var_value}')