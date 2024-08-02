from utils import generate_func, read_prompt_list

import opendit
from opendit import OpenSoraConfig, OpenSoraPipeline


def eval_base(prompt_list):
    config = OpenSoraConfig()
    pipeline = OpenSoraPipeline(config)

    generate_func(pipeline, prompt_list, "./samples/eval_base")


def eval_pab(prompt_list):
    config = OpenSoraConfig(enable_pab=True)
    pipeline = OpenSoraPipeline(config)

    generate_func(pipeline, prompt_list, "./samples/eval_pab")


if __name__ == "__main__":
    opendit.initialize(42)
    prompt_list = read_prompt_list("vbench/VBench_full_info_test.json")
    eval_base(prompt_list)
    eval_pab(prompt_list)
