import os

import opendit
from opendit import OpenSoraPlanConfig, OpenSoraPlanPipeline


# CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 scripts/opensora_plan/sample.py
def run_base():
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    print("Running base")
    opendit.initialize(42)

    config = OpenSoraPlanConfig()
    pipeline = OpenSoraPlanPipeline(config)

    for i in [1, 2]:
        print(f"Running iteration {i}")
        prompt = "a bear hunting for prey"
        video = pipeline.generate(prompt).video[0]
        pipeline.save_video(video, f"./outputs/opensora_plan_base_{prompt}.mp4")


def run_pab():
    # os.environ["RANK"] = "0"
    # os.environ["LOCAL_RANK"] = "0"
    # os.environ["WORLD_SIZE"] = "1"
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"
    print("Running PAB")
    opendit.initialize(42)

    config = OpenSoraPlanConfig(enable_pab=True)
    pipeline = OpenSoraPlanPipeline(config)

    prompt = "a bear hunting for prey"
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/opensora_plan_pab_{prompt}.mp4")


if __name__ == "__main__":
    run_base()  # 03:00
    # run_pab()  # 02:12

# CUDA_VISIBLE_DEVICES=3 torchrun --standalone --nproc_per_node=1 scripts/opensora_plan/sample.py
