from utils import generate_func, load_eval_prompts

from videosys import OpenSoraPlanConfig, OpenSoraPlanV110PABConfig, VideoSysEngine


def eval_base(prompt_list):
    config = OpenSoraPlanConfig(version="v110", transformer_type="65x512x512")
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/opensoraplan_base")


def eval_pab1(prompt_list):
    pab_config = OpenSoraPlanV110PABConfig(
        spatial_range=2,
        temporal_range=4,
        cross_range=6,
    )
    config = OpenSoraPlanConfig(version="v110", transformer_type="65x512x512", enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/opensoraplan_pab1")


def eval_pab2(prompt_list):
    pab_config = OpenSoraPlanV110PABConfig(
        spatial_range=3,
        temporal_range=5,
        cross_range=7,
    )
    config = OpenSoraPlanConfig(version="v110", transformer_type="65x512x512", enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/opensoraplan_pab2")


def eval_pab3(prompt_list):
    pab_config = OpenSoraPlanV110PABConfig(
        spatial_range=5,
        temporal_range=7,
        cross_range=9,
    )
    config = OpenSoraPlanConfig(version="v110", transformer_type="65x512x512", enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/opensoraplan_pab3")


if __name__ == "__main__":
    prompt_list = load_eval_prompts("./datasets/webvid_selected.csv")
    eval_base(prompt_list)
    eval_pab1(prompt_list)
    eval_pab2(prompt_list)
    eval_pab3(prompt_list)
