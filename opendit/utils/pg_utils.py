from colossalai.cluster.process_group_mesh import ProcessGroupMesh
from torch.distributed import ProcessGroup


class ProcessGroupManager(ProcessGroupMesh):
    def __init__(self, *size: int, dp_axis, sp_axis):
        super().__init__(*size)
        self.dp_axis = dp_axis
        self.sp_axis = sp_axis
        self._dp_group: ProcessGroup = self.get_group_along_axis(self.dp_axis)
        self._sp_group: ProcessGroup = self.get_group_along_axis(self.sp_axis)

    @property
    def dp_group(self) -> ProcessGroup:
        return self._dp_group

    @property
    def sp_group(self) -> ProcessGroup:
        return self._sp_group
