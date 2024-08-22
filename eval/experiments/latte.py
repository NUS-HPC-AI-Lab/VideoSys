from utils import generate_func, read_prompt_list

import opendit
from opendit import LatteConfig, LattePipeline
from opendit.models.latte import LattePABConfig


def eval_base(prompt_list):
    config = LatteConfig()
    pipeline = LattePipeline(config)

    generate_func(pipeline, prompt_list, "./samples/latte_base", loop=5)


def eval_pab1(prompt_list):
    pab_config = LattePABConfig(
        spatial_gap=2,
        temporal_gap=3,
        cross_gap=6,
    )
    config = LatteConfig(enable_pab=True, pab_config=pab_config)
    pipeline = LattePipeline(config)

    generate_func(pipeline, prompt_list, "./samples/latte_pab1", loop=5)


def eval_pab2(prompt_list):
    pab_config = LattePABConfig(
        spatial_gap=3,
        temporal_gap=4,
        cross_gap=7,
    )
    config = LatteConfig(enable_pab=True, pab_config=pab_config)
    pipeline = LattePipeline(config)

    generate_func(pipeline, prompt_list, "./samples/latte_pab2", loop=5)


def eval_pab3(prompt_list):
    pab_config = LattePABConfig(
        spatial_gap=4,
        temporal_gap=6,
        cross_gap=9,
    )
    config = LatteConfig(enable_pab=True, pab_config=pab_config)
    pipeline = LattePipeline(config)

    generate_func(pipeline, prompt_list, "./samples/latte_pab3", loop=5)


if __name__ == "__main__":
    opendit.initialize(42)
    prompt_list = read_prompt_list("vbench/VBench_full_info.json")
    eval_base(prompt_list)
    eval_pab1(prompt_list)
    eval_pab2(prompt_list)
    eval_pab3(prompt_list)
