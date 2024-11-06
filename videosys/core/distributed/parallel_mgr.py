import logging
import os
from collections import deque
from datetime import timedelta

import torch
import torch.distributed as dist
from colossalai.cluster.process_group_mesh import ProcessGroupMesh
from torch.distributed import ProcessGroup

from videosys.utils.logging import init_logger


class ParallelManager(ProcessGroupMesh):
    def __init__(self, dp_size, cp_size, sp_size):
        super().__init__(dp_size, cp_size, sp_size)
        dp_axis, cp_axis, sp_axis = 0, 1, 2
        self.switch_sp_cp = False

        self.dp_size = dp_size
        self.dp_group: ProcessGroup = self.get_group_along_axis(dp_axis)
        self.dp_rank = dist.get_rank(self.dp_group)

        self.cp_size = cp_size
        if cp_size > 1:
            self.cp_group: ProcessGroup = self.get_group_along_axis(cp_axis)
            self.cp_rank = dist.get_rank(self.cp_group)
        else:
            self.cp_group = None
            self.cp_rank = None

        self.sp_size = sp_size
        if sp_size > 1:
            self.sp_group: ProcessGroup = self.get_group_along_axis(sp_axis)
            self.sp_rank = dist.get_rank(self.sp_group)
        else:
            self.sp_group = None
            self.sp_rank = None

        logging.info(f"Init parallel manager with dp_size: {dp_size}, cp_size: {cp_size}, sp_size: {sp_size}")


class DynamicParallelManager:
    def __init__(self, dp_size, enable_gas_queue=False):
        self._rank = dist.get_rank()
        self.dp_group: ProcessGroup = dist.new_group(list(range(0, dist.get_world_size())))
        self.dp_rank = dist.get_rank(self.dp_group)

        self.sp_rank = None
        self.sp_size = None
        self.sp_group = None
        self.gloo_sp_group = None
        self.enable_sp = False
        self.cp_size = 1
        self.switch_sp_cp = False

        # {sp_size: sp_group}
        self.sp_clusters = {}
        # for monitoring hangs by nccl internal error
        self.gloo_sp_clusters = {}
        self._build_clusters()
        self.sp_queue = deque()
        self.sync_tensor = torch.tensor(
            [0, 0, 0], dtype=torch.int, device=torch.device(f"cuda:{torch.cuda.current_device()}")
        )

    def _build_clusters(self):
        wsize = dist.get_world_size()
        _s = 1
        global_ranks = list(range(0, wsize))
        while _s <= wsize:
            group_start_indices = list(range(0, wsize, _s))
            for group_start_idx in group_start_indices:
                group_ranks = global_ranks[group_start_idx : group_start_idx + _s]
                gpu_group = dist.new_group(group_ranks, use_local_synchronization=True, timeout=timedelta(seconds=60))
                cpu_group = dist.new_group(group_ranks, backend="gloo", use_local_synchronization=True)
                if self._rank in group_ranks:
                    # dist.distributed_c10d._world.pg_default_device[gpu_group] = torch.device(f'cuda:{torch.cuda.current_device()}')
                    self.sp_clusters[_s] = gpu_group
                    self.gloo_sp_clusters[_s] = cpu_group
            _s *= 2

    def set_sp_size(self, sp_size):
        if sp_size == self.sp_size:
            return
        self.sp_size = sp_size
        self.sp_group = self.sp_clusters[sp_size]
        self.gloo_sp_group = self.gloo_sp_clusters[sp_size]
        self.sp_rank = dist.get_rank(self.sp_group)
        self.enable_sp = sp_size > 1

        torch.cuda.ipc_collect()
        # use all reduce to sync instead of barrier, as barrier with small sp groups will launch multiple process in other gpus
        # dist.all_reduce(self.sync_tensor, group=self.sp_group)

    def append_sp_size(self, sp_size):
        self.sp_queue.append(sp_size)

    def pop_sp_size(self):
        return self.sp_queue.popleft()


def initialize(
    rank=0,
    world_size=1,
    init_method=None,
):
    if not dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception:
            pass
        dist.init_process_group(backend="nccl", init_method=init_method, world_size=world_size, rank=rank)
        torch.cuda.set_device(rank)
        init_logger()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def set_distributed_state(distributed_profile=None):
    # OMPI_* assume for OpenMPI, RANK/WORLD_SIZE assume torchrun
    rank = int(os.getenv("RANK", os.getenv("OMPI_COMM_WORLD_RANK", "-1")))
    world_size = int(os.getenv("WORLD_SIZE", os.getenv("OMPI_COMM_WORLD_SIZE", "-1")))
    device_count = torch.cuda.device_count()
    node_rank = int(os.getenv("NODE_RANK", os.getenv("OMPI_COMM_WORLD_NODE_RANK", "0")))
    node_size = int(os.getenv("NNODES", "1"))

    if distributed_profile:
        "launch multiple single-node instances for fast profile"
        assert world_size % device_count == 0
        node_rank = rank // device_count
        node_size = world_size // device_count
        new_rank = rank % device_count
        new_world_size = device_count

        os.environ["NNODES"] = "1"
        os.environ["NODE_RANK"] = "0"
        os.environ["RANK"] = str(new_rank)
        os.environ["WORLD_SIZE"] = str(new_world_size)
        os.environ["MASTER_ADDR"] = "localhost"
        print(
            f">>> [Distributed Profile] detect {node_size} for fast profile. Rank: {rank}/{world_size} overwrite to {new_rank}/{new_world_size} env: {os.environ['RANK']}/{os.environ['WORLD_SIZE']}"
        )

        rank = new_rank
        world_size = new_world_size

    return rank, world_size, node_rank, node_size
