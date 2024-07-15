import colossalai
import torch
import torch.distributed as dist
from colossalai.cluster.process_group_mesh import ProcessGroupMesh
from torch.distributed import ProcessGroup

from opendit.utils.utils import set_seed

PARALLEL_MANAGER = None


class ParallelManager(ProcessGroupMesh):
    def __init__(self, dp_size, sp_size, dp_axis, sp_axis):
        super().__init__(dp_size, sp_size)
        self.dp_axis = dp_axis
        self.dp_group: ProcessGroup = self.get_group_along_axis(self.dp_axis)
        self.dp_rank = dist.get_rank(self.dp_group)

        self.sp_size = sp_size
        self.sp_axis = sp_axis
        self.sp_group: ProcessGroup = self.get_group_along_axis(self.sp_axis)
        self.sp_rank = dist.get_rank(self.sp_group)
        self.enable_sp = sp_size > 1


def set_parallel_manager(dp_size, sp_size, dp_axis=0, sp_axis=1):
    global PARALLEL_MANAGER
    PARALLEL_MANAGER = ParallelManager(dp_size, sp_size, dp_axis, sp_axis)


def get_data_parallel_group():
    return PARALLEL_MANAGER.dp_group


def get_data_parallel_rank():
    return PARALLEL_MANAGER.dp_rank


def get_sequence_parallel_group():
    return PARALLEL_MANAGER.sp_group


def get_sequence_parallel_size():
    return PARALLEL_MANAGER.sp_size


def get_sequence_parallel_rank():
    return PARALLEL_MANAGER.sp_rank


def enable_sequence_parallel():
    if PARALLEL_MANAGER is None:
        return False
    return PARALLEL_MANAGER.enable_sp


def get_parallel_manager():
    return PARALLEL_MANAGER


def initialize(seed: int = None, sp_size: int = None):
    colossalai.launch_from_torch({})
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if sp_size is None:
        sp_size = dist.get_world_size()
        dp_size = 1
    else:
        assert dist.get_world_size() % sp_size == 0, f"world_size {dist.get_world_size()} must be divisible by sp_size"
        dp_size = dist.get_world_size() // sp_size

    set_parallel_manager(dp_size, sp_size)

    if seed is not None:
        local_seed = seed + get_data_parallel_rank()
        set_seed(local_seed)
