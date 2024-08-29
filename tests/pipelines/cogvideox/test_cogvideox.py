import pytest

from videosys import CogVideoXConfig, VideoSysEngine


@pytest.mark.parametrize("num_gpus", [1])
def test_base(num_gpus):
    config = CogVideoXConfig(num_gpus=num_gpus)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./test_outputs/{prompt}_cogvideo_{num_gpus}.mp4")


@pytest.mark.parametrize("num_gpus", [1])
def test_pab(num_gpus):
    config = CogVideoXConfig(num_gpus=num_gpus, enable_pab=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./test_outputs/{prompt}_cogvideo_pab_{num_gpus}.mp4")
