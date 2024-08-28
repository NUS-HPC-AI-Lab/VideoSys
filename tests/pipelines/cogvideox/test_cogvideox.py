import pytest

from videosys import CogVideoXConfig, VideoSysEngine


@pytest.mark.parametrize("world_size", [1])
def test_base(world_size):
    config = CogVideoXConfig(world_size=world_size)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./test_outputs/{prompt}_cogvideo_{world_size}.mp4")


@pytest.mark.parametrize("world_size", [1])
def test_pab(world_size):
    config = CogVideoXConfig(world_size=world_size, enable_pab=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./test_outputs/{prompt}_cogvideo_pab_{world_size}.mp4")
