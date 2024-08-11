import os

import deltadit
from deltadit import OpenSoraPlanConfig, OpenSoraPlanPipeline


def run_base():
    # Manually set environment variables for single GPU debugging
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    deltadit.initialize(42)

    # --gate_step 15 \
    # --sa_interval 3 \
    # --ca_interval 1 \

    config = OpenSoraPlanConfig()
    pipeline = OpenSoraPlanPipeline(config)

    prompt = "Sunset over the sea."
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

    config = OpenSoraPlanConfig(enable_pab=True)
    pipeline = OpenSoraPlanPipeline(config)

    prompt = "Sunset over the sea."
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/{prompt}.mp4")


if __name__ == "__main__":
    # run_base()
    run_pab()
