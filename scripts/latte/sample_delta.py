# Usage: torchrun --standalone --nproc_per_node=1 scripts/latte/sample.py

import os

import deltadit
from deltadit import LatteConfig, LatteDELTAConfig, LattePipeline


def run_base():
    # Manually set environment variables for single GPU debugging
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    deltadit.initialize(42)

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

    deltadit.initialize(42)

    delta_config = LatteDELTAConfig(
        steps=10,
        delta_skip=False,
        # delta_threshold={(0, 10): [0, 9]},
        delta_threshold={(3, 4): [1, 2]},
        # delta_threshold={(0,10):[0,9], (20,30):[9,27]},
        # delta_threshold={(0,10):[9,27], (20,30):[9,27]},
        delta_gap=2,
    )
    # step 250 / m=100 / k=10
    # opensora step=30 / m=12 / k=2
    # latte step=50 / m=20 / k=2
    config = LatteConfig(enable_delta=True, delta_config=delta_config)
    pipeline = LattePipeline(config)

    prompt = "Yellow and black tropical fish dart through the sea."
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/{prompt}.mp4")


if __name__ == "__main__":
    # run_base()
    run_pab()
