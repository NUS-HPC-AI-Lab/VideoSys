# Usage: torchrun --standalone --nproc_per_node=1 scripts/latte/sample.py

import os

import tgate
from tgate import LatteConfig, LattePipeline, LatteTGATEConfig


def run_base():
    # Manually set environment variables for single GPU debugging
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    tgate.initialize(42)

    config = LatteConfig()
    pipeline = LattePipeline(config)

    prompt = "Yellow and black tropical fish dart through the sea."
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/{prompt}.mp4")


def run_pab():
    # Manually set environment variables for single GPU debugging
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    tgate.initialize(42)

    tgate_config = LatteTGATEConfig(
        spatial_broadcast=True,
        spatial_threshold=[0, 20],
        spatial_gap=2,
        temporal_broadcast=True,
        temporal_threshold=[0, 20],
        temporal_gap=2,
        cross_broadcast=True,
        cross_threshold=[20, 50],
        cross_gap=20,
    )
    # step 250 / m=100 / k=10
    # opensora step=30 / m=12 / k=2
    # latte step=50 / m=20 / k=2
    config = LatteConfig(enable_tgate=True, tgate_config=tgate_config)
    pipeline = LattePipeline(config)

    prompt = "Yellow and black tropical fish dart through the sea."
    video = pipeline.generate(prompt).video[0]

    save_path = f"./outputs/latte_{prompt.replace(' ', '_')}_spatial_{config.tgate_config.spatial_threshold}_cross_{config.tgate_config.cross_threshold}_tgate.mp4"
    pipeline.save_video(video, save_path)
    print(f"Saved video to {save_path}")


if __name__ == "__main__":
    # run_base()
    run_pab()
