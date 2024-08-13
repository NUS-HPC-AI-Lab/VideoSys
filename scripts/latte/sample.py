# Usage: torchrun --standalone --nproc_per_node=1 scripts/latte/sample.py

import opendit
from opendit import LatteConfig, LattePipeline


def run_base():
    opendit.initialize(42)

    config = LatteConfig()
    pipeline = LattePipeline(config)

    prompt = "a bear hunting for prey"
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/{prompt}.mp4")


def run_pab():
    opendit.initialize(42)

    config = LatteConfig(enable_pab=True)
    pipeline = LattePipeline(config)

    prompt = "a bear hunting for prey"
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/{prompt}.mp4")


if __name__ == "__main__":
    # run_base()
    run_pab()
