import os

import opendit
from opendit import OpenSoraPlanConfig, OpenSoraPlanPipeline


# torchrun --standalone --nproc_per_node=1 scripts/opensora_plan/sample.py
def run_base():
    opendit.initialize(42)

    config = OpenSoraPlanConfig()
    pipeline = OpenSoraPlanPipeline(config)

    prompt = "a bear hunting for prey"
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/{prompt}.mp4")


def run_pab():
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    opendit.initialize(42)

    config = OpenSoraPlanConfig(enable_pab=True)
    pipeline = OpenSoraPlanPipeline(config)

    prompt = "a bear hunting for prey"
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/{prompt}.mp4")


if __name__ == "__main__":
    # run_base() 03:39
    run_pab()  # 02:51
