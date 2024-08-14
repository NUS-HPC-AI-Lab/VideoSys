import opendit
from opendit import OpenSoraConfig, OpenSoraPipeline


def run_base():
    opendit.initialize(42)

    config = OpenSoraConfig()
    pipeline = OpenSoraPipeline(config)

    prompt = "a bear hunting for prey"
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/opensora_base_{prompt}.mp4")


def run_pab():
    opendit.initialize(42)

    config = OpenSoraConfig(enable_pab=True)
    pipeline = OpenSoraPipeline(config)

    prompt = "a bear hunting for prey"
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/opensora_pab_{prompt}.mp4")


if __name__ == "__main__":
    # run_base() # 01:18
    run_pab()  # 01:01

# CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 scripts/opensora/sample.py
