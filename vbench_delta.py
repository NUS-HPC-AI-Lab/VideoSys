from utils import generate_func, read_prompt_list
import deltadit
from deltadit import LatteConfig, LatteDELTAConfig, LattePipeline, OpenSoraConfig, OpenSoraDELTAConfig, OpenSoraPipeline, OpenSoraPlanDELTAConfig, OpenSoraPlanConfig, OpenSoraPlanPipeline


def eval_opensora(prompt_list):
    delta_config = OpenSoraDELTAConfig(
        steps=10,
        delta_skip=True,
        delta_threshold={(0, 5): [0, 5]},
        delta_gap=2,
    )
    config = OpenSoraConfig(enable_delta=True, delta_config=delta_config)
    pipeline = OpenSoraPipeline(config)
    generate_func(pipeline, prompt_list, "./samples/delta_opensora")


def eval_latte(prompt_list):
    delta_config = LatteDELTAConfig(
        steps=10,
        delta_skip=True,
        delta_threshold={(0, 1): [0, 1]},
        delta_gap=2,
    )
    config = LatteConfig(enable_delta=True, delta_config=delta_config)
    pipeline = LattePipeline(config)
    generate_func(pipeline, prompt_list, "./samples/delta_latte")


def eval_opensora_plan(prompt_list):
    delta_config = OpenSoraPlanDELTAConfig(
        steps=10,
        delta_skip=True,
        delta_threshold={(0, 1): [0, 2]},
        delta_gap=2,
    )
    config = OpenSoraPlanConfig(enable_delta=True, delta_config=delta_config)
    pipeline = OpenSoraPlanPipeline(config)
    generate_func(pipeline, prompt_list, "./samples/delta_opensora_plan")


if __name__ == "__main__":
    deltadit.initialize(42)
    prompt_list = read_prompt_list("./VBench_full_info.json")
    # eval_opensora(prompt_list)
    # eval_latte(prompt_list)
    eval_opensora_plan(prompt_list)
