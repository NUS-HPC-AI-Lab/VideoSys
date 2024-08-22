from utils import generate_func, read_prompt_list

import videosys
from videosys import OpenSoraConfig, OpenSoraPipeline
from videosys.models.open_sora import OpenSoraPABConfig


def eval_base(prompt_list):
    config = OpenSoraConfig()
    pipeline = OpenSoraPipeline(config)

    generate_func(pipeline, prompt_list, "./samples/opensora_base", loop=5)


def eval_pab1(prompt_list):
    config = OpenSoraConfig(enable_pab=True)
    pipeline = OpenSoraPipeline(config)

    generate_func(pipeline, prompt_list, "./samples/opensora_pab1", loop=5)


def eval_pab2(prompt_list):
    pab_config = OpenSoraPABConfig(spatial_gap=3, temporal_gap=5, cross_gap=7)
    config = OpenSoraConfig(enable_pab=True, pab_config=pab_config)
    pipeline = OpenSoraPipeline(config)

    generate_func(pipeline, prompt_list, "./samples/opensora_pab2", loop=5)


def eval_pab3(prompt_list):
    pab_config = OpenSoraPABConfig(spatial_gap=5, temporal_gap=7, cross_gap=9)
    config = OpenSoraConfig(enable_pab=True, pab_config=pab_config)
    pipeline = OpenSoraPipeline(config)

    generate_func(pipeline, prompt_list, "./samples/opensora_pab3", loop=5)


if __name__ == "__main__":
    videosys.initialize(42)
    prompt_list = read_prompt_list("vbench/VBench_full_info.json")
    eval_base(prompt_list)
    eval_pab1(prompt_list)
    eval_pab2(prompt_list)
    eval_pab3(prompt_list)
