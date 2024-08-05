from utils import generate_func, read_prompt_list

import opendit
from opendit import LatteConfig, LattePipeline


def eval_base(prompt_list):
    config = LatteConfig()
    pipeline = LattePipeline(config)

    generate_func(pipeline, prompt_list, "./samples/latte_base", loop=1)


def eval_pab(prompt_list):
    config = LatteConfig(enable_pab=True)
    pipeline = LattePipeline(config)

    generate_func(pipeline, prompt_list, "./samples/latte_pab", loop=1)


if __name__ == "__main__":
    opendit.initialize(42)
    prompt_list = read_prompt_list("vbench/VBench_full_info_test.json")
    eval_base(prompt_list)
    eval_pab(prompt_list)
