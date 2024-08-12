import os

import torch

import tgate
from tgate import OpenSoraConfig, OpenSoraPipeline, OpenSoraTGATEConfig


def run_base():
    # Manually set environment variables for single GPU debugging
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    tgate.initialize(42)

    config = OpenSoraConfig()
    pipeline = OpenSoraPipeline(config)

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
    # step 250 / m=100 / k=10
    # opensora step=30 / m=12 / k=2
    # latte step=50 / m=20 / k=2
    config = OpenSoraConfig(enable_tgate=True, tgate_config=tgate_config)
    pipeline = OpenSoraPipeline(config)

    prompt = "Yellow and black tropical fish dart through the sea."
    video = pipeline.generate(prompt).video[0]

    save_path = f"./outputs/opensora_{prompt.replace(' ', '_')}_spatial_{config.tgate_config.spatial_threshold}_cross_{config.tgate_config.cross_threshold}_tgate.mp4"
    pipeline.save_video(video, save_path)
    print(f"Saved video to {save_path}")


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False

    # run_base()
    run_pab()
