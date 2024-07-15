import opendit
from opendit import LatteConfig, LattePipeline


def run_base():
    opendit.initialize(42)

    config = LatteConfig()
    pipeline = LattePipeline(config)

    prompt = "Sunset over the sea."
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/{prompt[:50]}.mp4")


def run_pab():
    opendit.initialize(42)

    config = LatteConfig(enable_pab=True)
    pipeline = LattePipeline(config)

    prompt = "Sunset over the sea."
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/{prompt[:50]}.mp4")


if __name__ == "__main__":
    run_base()
    # run_pab()
