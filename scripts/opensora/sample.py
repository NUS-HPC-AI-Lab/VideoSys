import opendit
from opendit import OpenSoraConfig, OpenSoraPipeline
from opendit.core.engine import VideoSysEngine


def run_pab(rank=0, world_size=1, init_method=None):
    opendit.initialize(rank, world_size, init_method, 42)

    config = OpenSoraConfig(enable_pab=True)
    pipeline = OpenSoraPipeline(config)

    prompt = "Sunset over the sea."
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/{prompt}.mp4")


def run_base():
    config = OpenSoraConfig(world_size=2)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


def run_pab():
    config = OpenSoraConfig(world_size=2, enable_pab=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


if __name__ == "__main__":
    run_base()
    run_pab()
