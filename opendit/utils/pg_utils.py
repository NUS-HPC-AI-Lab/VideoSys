from colossalai.cluster.process_group_mesh import ProcessGroupMesh
from torch import nn
from torch.distributed import ProcessGroup


class ProcessGroupManager(ProcessGroupMesh):
    def __init__(self, *size: int, dp_axis, sp_axis):
        super().__init__(*size)
        self.dp_axis = dp_axis
        self.sp_axis = sp_axis
        self._dp_group = self.get_group_along_axis(self.dp_axis)
        self._sp_group = self.get_group_along_axis(self.sp_axis)

    @property
    def dp_group(self):
        return self._dp_group

    @property
    def get_sp_group(self):
        return self._sp_group


# Register sequence parallel group after copy
def register_sequence_parallel_group(model: nn.Module, sp_group: ProcessGroup):
    setattr(model, "sequence_parallel_group", sp_group)
    for block in model.blocks:
        setattr(block, "sequence_parallel_group", sp_group)
        setattr(block.attn, "sequence_parallel_group", sp_group)
    return model
