import opendit
from opendit import OpenSoraConfig, OpenSoraPipeline
from opendit.core.engine import VideoSysEngine


def run_base(rank=0, world_size=1, init_method=None):
    opendit.initialize(rank, world_size, init_method, 42)

    config = OpenSoraConfig()
    pipeline = OpenSoraPipeline(config)

    prompt = "Sunset over the sea."
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/{prompt}.mp4")


def run_pab(rank=0, world_size=1, init_method=None):
    opendit.initialize(rank, world_size, init_method, 42)

    config = OpenSoraConfig(enable_pab=True)
    pipeline = OpenSoraPipeline(config)

    prompt = "Sunset over the sea."
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/{prompt}.mp4")


def run(world_size=1):
    config = OpenSoraConfig(world_size)
    engine = VideoSysEngine(OpenSoraPipeline, config)
    # pipeline = OpenSoraPipeline(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")
    # engine.shutdown()


if __name__ == "__main__":
    # run_base()
    # run_pab()
    run()
