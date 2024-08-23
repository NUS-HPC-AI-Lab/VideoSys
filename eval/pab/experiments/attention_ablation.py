from utils import generate_func, read_prompt_list

import videosys
from videosys import OpenSoraConfig, OpenSoraPipeline
from videosys.models.open_sora import OpenSoraPABConfig


def attention_ablation_func(pab_kwargs, prompt_list, output_dir):
    pab_config = OpenSoraPABConfig(**pab_kwargs)
    config = OpenSoraConfig(enable_pab=True, pab_config=pab_config)
    pipeline = OpenSoraPipeline(config)

    generate_func(pipeline, prompt_list, output_dir)


def main(prompt_list):
    # spatial
    gap_list = [2, 3, 4, 5]
    for gap in gap_list:
        pab_kwargs = {
            "spatial_broadcast": True,
            "spatial_gap": gap,
            "temporal_broadcast": False,
            "cross_broadcast": False,
            "mlp_skip": False,
        }
        output_dir = f"./samples/attention_ablation/spatial_g{gap}"
        attention_ablation_func(pab_kwargs, prompt_list, output_dir)

    # temporal
    gap_list = [3, 4, 5, 6]
    for gap in gap_list:
        pab_kwargs = {
            "spatial_broadcast": False,
            "temporal_broadcast": True,
            "temporal_gap": gap,
            "cross_broadcast": False,
            "mlp_skip": False,
        }
        output_dir = f"./samples/attention_ablation/temporal_g{gap}"
        attention_ablation_func(pab_kwargs, prompt_list, output_dir)

    # cross
    gap_list = [5, 6, 7, 8]
    for gap in gap_list:
        pab_kwargs = {
            "spatial_broadcast": False,
            "temporal_broadcast": False,
            "cross_broadcast": True,
            "cross_gap": gap,
            "mlp_skip": False,
        }
        output_dir = f"./samples/attention_ablation/cross_g{gap}"
        attention_ablation_func(pab_kwargs, prompt_list, output_dir)


if __name__ == "__main__":
    videosys.initialize(42)
    prompt_list = read_prompt_list("vbench/VBench_full_info.json")
    main(prompt_list)
