import videosys
from videosys import CogVideoConfig, CogVideoPipeline


def run_base():
    videosys.initialize(42)

    config = CogVideoConfig()
    pipeline = CogVideoPipeline(config)

    prompt = "A cat swimming"
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/{prompt}.mp4")


if __name__ == "__main__":
    run_base()
