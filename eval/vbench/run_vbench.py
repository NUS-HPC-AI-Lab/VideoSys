import torch
from vbench import VBench

full_info_path = "./vbench/VBench_test.json"

dimensions = [
    "subject_consistency",
    "imaging_quality",
    "background_consistency",
    "motion_smoothness",
    "overall_consistency",
    "human_action",
    "multiple_objects",
    "spatial_relationship",
    "object_class",
    "color",
    "aesthetic_quality",
    "appearance_style",
    "temporal_flickering",
    "scene",
    "temporal_style",
    "dynamic_degree",
]

for dimension in dimensions:
    my_VBench = VBench(torch.device("cuda"), full_info_path, "vbench_out")
    my_VBench.evaluate(
        videos_path="./videos",
        name=dimension,
        local=False,
        read_frame=False,
        dimension_list=[dimension],
        mode="vbench_standard",
    )
