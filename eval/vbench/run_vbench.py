import torch
from vbench import VBench

full_info_path = "/data/xuanlei/Lumiere-exp/eval/vbench/VBench_test.json"


my_VBench = VBench(torch.device("cuda"), full_info_path, "vbench_out")
my_VBench.evaluate(
    videos_path="/data/xuanlei/Lumiere-exp/eval/videos",
    name="temporal_flickering",
    local=False,
    read_frame=False,
    dimension_list=["temporal_flickering"],
    mode="vbench_standard",
)
