from utils import generate_func, read_prompt_list
import tgate
from tgate import OpenSoraConfig, OpenSoraPipeline, OpenSoraTGATEConfig, LatteConfig, LattePipeline, LatteTGATEConfig,OpenSoraPlanConfig, OpenSoraPlanPipeline, OpenSoraPlanTGATEConfig


def eval_opensora(prompt_list):
    tgate_config = OpenSoraTGATEConfig(
        spatial_broadcast=True,
        spatial_threshold=[0, 12],
        spatial_gap=2,
        temporal_broadcast=True,
        temporal_threshold=[0, 12],
        temporal_gap=2,
        cross_broadcast=True,
        cross_threshold=[12, 30],
        cross_gap=12,
    )
    config = OpenSoraConfig(enable_tgate=True, tgate_config=tgate_config)
    pipeline = OpenSoraPipeline(config)
    generate_func(pipeline, prompt_list, "./samples/tgate_opensora")


def eval_opensoraplan(prompt_list):
    tgate_config = OpenSoraPlanTGATEConfig(
        spatial_broadcast=True,
        spatial_threshold=[0, 90],
        spatial_gap=6,
        temporal_broadcast=True,
        temporal_threshold=[0, 90],
        temporal_gap=6,
        cross_broadcast=True,
        cross_threshold=[90, 150],
        cross_gap=60,
    )
    config = OpenSoraPlanConfig(enable_tgate=False, tgate_config=tgate_config)
    pipeline = OpenSoraPlanPipeline(config)
    generate_func(pipeline, prompt_list, "./samples/tgate_opensoraplan")



if __name__ == "__main__":
    tgate.initialize(42)
    prompt_list = read_prompt_list("./VBench_full_info.json")
    eval_opensora(prompt_list)
    eval_opensoraplan(prompt_list)
