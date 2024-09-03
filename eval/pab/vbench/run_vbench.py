import argparse

import torch
from vbench import VBench

full_info_path = "./vbench/VBench_full_info.json"

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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, type=str)
    parser.add_argument("--save_path", required=True, type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    kwargs = {}
    kwargs["imaging_quality_preprocessing_mode"] = "longer"  # use VBench/evaluate.py default

    for dimension in dimensions:
        my_VBench = VBench(torch.device("cuda"), full_info_path, args.save_path)
        my_VBench.evaluate(
            videos_path=args.video_path,
            name=dimension,
            local=False,
            read_frame=False,
            dimension_list=[dimension],
            mode="vbench_standard",
            **kwargs,
        )
