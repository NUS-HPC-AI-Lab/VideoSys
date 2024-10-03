from utils import generate_func, load_eval_prompts

from videosys import LatteConfig, LattePABConfig, VideoSysEngine


def eval_base(prompt_list):
    config = LatteConfig()
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/latte_base")


def eval_pab1(prompt_list):
    pab_config = LattePABConfig(
        spatial_range=2,
        temporal_range=3,
        cross_range=6,
    )
    config = LatteConfig(enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/latte_pab1")


def eval_pab2(prompt_list):
    pab_config = LattePABConfig(
        spatial_range=3,
        temporal_range=4,
        cross_range=7,
    )
    config = LatteConfig(enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/latte_pab2")


def eval_pab3(prompt_list):
    pab_config = LattePABConfig(
        spatial_range=4,
        temporal_range=6,
        cross_range=9,
    )
    config = LatteConfig(enable_pab=True, pab_config=pab_config)
    engine = VideoSysEngine(config)
    generate_func(engine, prompt_list, "./samples/latte_pab3")


if __name__ == "__main__":
    prompt_list = load_eval_prompts("./datasets/webvid_selected.csv")
    eval_base(prompt_list)
    eval_pab1(prompt_list)
    eval_pab2(prompt_list)
    eval_pab3(prompt_list)
