import pytest

from videosys import CogVideoConfig, VideoSysEngine


@pytest.mark.parametrize("world_size", [1, 2])
def test_base(world_size):
    config = CogVideoConfig(world_size)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}_cogvideo_{world_size}.mp4")
