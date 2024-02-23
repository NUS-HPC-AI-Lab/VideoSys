from colossalai.cluster.process_group_mesh import ProcessGroupMesh
from torch import nn
from torch.distributed import ProcessGroup

DP_AXIS = 0
SP_AXIS = 1


class ProcessGroupManager(ProcessGroupMesh):
    def __init__(self, *size: int):
        super().__init__(*size)


# Initialize a process group manager
def initialize_process_group_maneger(world_size: int, sp_size: int) -> ProcessGroupManager:
    assert world_size % sp_size == 0, f"World size {world_size} is not divisible by sequence parallel size {sp_size}"
    dp_size = world_size // sp_size
    pg_manager = ProcessGroupManager(dp_size, sp_size)
    return pg_manager


# Register sequence parallel group after copy
def register_sequence_parallel_group(model: nn.Module, sp_group: ProcessGroup):
    setattr(model, "sequence_parallel_group", sp_group)
    for block in model.blocks:
        setattr(block, "sequence_parallel_group", sp_group)
        setattr(block.attn, "sequence_parallel_group", sp_group)
    return model
