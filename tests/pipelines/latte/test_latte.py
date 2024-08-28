import pytest

from videosys import LatteConfig, VideoSysEngine


@pytest.mark.parametrize("world_size", [1, 2])
def test_base(world_size):
    config = LatteConfig(world_size=world_size)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./test_outputs/{prompt}_latte_{world_size}.mp4")


@pytest.mark.parametrize("world_size", [1, 2])
def test_pab(world_size):
    config = LatteConfig(world_size=world_size)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./test_outputs/{prompt}_latte_pab_{world_size}.mp4")
