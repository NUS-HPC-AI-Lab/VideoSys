from colossalai.cluster.process_group_mesh import ProcessGroupMesh
from torch.distributed import ProcessGroup

PARALLEL_MANAGER = None


class ParallelManager(ProcessGroupMesh):
    def __init__(self, dp_size, sp_size, dp_axis, sp_axis):
        super().__init__(dp_size, sp_size)
        self.dp_axis = dp_axis
        self.sp_axis = sp_axis
        self.dp_group: ProcessGroup = self.get_group_along_axis(self.dp_axis)
        self.sp_group: ProcessGroup = self.get_group_along_axis(self.sp_axis)


def set_parallel_manager(dp_size, sp_size, dp_axis, sp_axis):
    global PARALLEL_MANAGER
    PARALLEL_MANAGER = ParallelManager(dp_size, sp_size, dp_axis, sp_axis)


def get_sequence_parallel_group():
    return PARALLEL_MANAGER.sp_group


def get_data_parallel_group():
    return PARALLEL_MANAGER.dp_group


def get_parallel_manager():
    return PARALLEL_MANAGER
