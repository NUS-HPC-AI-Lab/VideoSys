import os

import tgate
from tgate import OpenSoraPlanConfig, OpenSoraPlanPipeline, OpenSoraPlanTGATEConfig


def run_base():
    # Manually set environment variables for single GPU debugging
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12358"
    print("Running tgate base")
    tgate.initialize(42)

    # --gate_step 15 \
    # --sa_interval 3 \
    # --ca_interval 1 \

    config = OpenSoraPlanConfig()
    pipeline = OpenSoraPlanPipeline(config)

    for i in [1]:
        print(f"Running iteration {i}")
        prompt = "a bear hunting for prey"
        video = pipeline.generate(prompt).video[0]
        pipeline.save_video(video, f"./outputs/opensora_plan_tgate_base_{prompt}.mp4")


def run_pab():
    # Manually set environment variables for single GPU debugging
    # os.environ["RANK"] = "0"
    # os.environ["LOCAL_RANK"] = "0"
    # os.environ["WORLD_SIZE"] = "1"
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12358"
    print("Running tgate pab")
    tgate.initialize(42)

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
    # step=150 / m=90 / k=6
    # step=30 / m=12 / k=2
    # step=50 / m=20 / k=2
    config = OpenSoraPlanConfig(enable_tgate=True, tgate_config=tgate_config)
    pipeline = OpenSoraPlanPipeline(config)

    prompt = "a bear hunting for prey"
    video = pipeline.generate(prompt).video[0]

    save_path = f"./outputs/opensora_plan_tgate_acc_{prompt.replace(' ', '_')}_spatial_{config.tgate_config.spatial_threshold}_cross_{config.tgate_config.cross_threshold}.mp4"
    pipeline.save_video(video, save_path)
    print(f"Saved video to {save_path}")


if __name__ == "__main__":
    # torch.backends.cudnn.enabled = False
    # run_base()  # 02:59
    run_pab()  # enable_tgate=True

# CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 scripts/opensora_plan/sample_tgate.py
