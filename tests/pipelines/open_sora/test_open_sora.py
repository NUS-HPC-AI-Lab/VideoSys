import pytest

from videosys import OpenSoraConfig, VideoSysEngine


@pytest.mark.parametrize("world_size", [1, 2])
def test_base(world_size):
    config = OpenSoraConfig(num_gpus=world_size)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./test_outputs/{prompt}_open_sora_{world_size}.mp4")


@pytest.mark.parametrize("world_size", [1, 2])
def test_pab(world_size):
    config = OpenSoraConfig(num_gpus=world_size, enable_pab=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./test_outputs/{prompt}_open_sora_pab_{world_size}.mp4")
