import videosys
from videosys import OpenSoraPlanConfig, OpenSoraPlanPipeline


def run_base():
    videosys.initialize(42)

    config = OpenSoraPlanConfig()
    pipeline = OpenSoraPlanPipeline(config)

    prompt = "Sunset over the sea."
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/{prompt}.mp4")


def run_pab():
    videosys.initialize(42)

    config = OpenSoraPlanConfig(enable_pab=True)
    pipeline = OpenSoraPlanPipeline(config)

    prompt = "Sunset over the sea."
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/{prompt}.mp4")


if __name__ == "__main__":
    run_base()
    # run_pab()
