from utils import generate_func, read_prompt_list

import opendit
from opendit import OpenSoraConfig, OpenSoraPipeline


def eval_base(prompt_list):
    config = OpenSoraConfig()
    pipeline = OpenSoraPipeline(config)

    generate_func(pipeline, prompt_list, "./samples/opensora_base", loop=1)


def eval_pab(prompt_list):
    config = OpenSoraConfig(enable_pab=True)
    pipeline = OpenSoraPipeline(config)

    generate_func(pipeline, prompt_list, "./samples/opensora_pab", loop=1)


if __name__ == "__main__":
    opendit.initialize(42)
    prompt_list = read_prompt_list("vbench/VBench_full_info.json")
    eval_base(prompt_list)
    eval_pab(prompt_list)
