# Usage: torchrun --standalone --nproc_per_node=1 scripts/latte/sample.py

import opendit
from opendit import LatteConfig, LattePipeline


def run_base():
    opendit.initialize(42)

    config = LatteConfig()
    pipeline = LattePipeline(config)

    prompt = "a bear hunting for prey"
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/latte_base_{prompt}.mp4")


def run_pab():
    opendit.initialize(42)

    config = LatteConfig(enable_pab=True)
    pipeline = LattePipeline(config)

    prompt = "a bear hunting for prey"
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/latte_pab_{prompt}.mp4")


if __name__ == "__main__":
    # run_base() # 00:46
    run_pab()  # 00:38

# CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 scripts/latte/sample.py
