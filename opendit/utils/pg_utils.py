from colossalai.cluster.process_group_mesh import ProcessGroupMesh

DP_AXIS = 0
SP_AXIS = 1


class ProcessGroupManager(ProcessGroupMesh):
    def __init__(self, *size: int):
        super().__init__(*size)
