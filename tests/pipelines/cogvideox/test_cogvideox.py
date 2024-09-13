import pytest

from videosys import CogVideoXConfig, VideoSysEngine


@pytest.mark.parametrize("num_gpus", [1, 2])
def test_base(num_gpus):
    config = CogVideoXConfig(num_gpus=num_gpus)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt, seed=0).video[0]
    engine.save_video(video, f"./test_outputs/{prompt}_cogvideo_{num_gpus}.mp4")


@pytest.mark.parametrize("num_gpus", [1])
def test_pab(num_gpus):
    config = CogVideoXConfig(num_gpus=num_gpus, enable_pab=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt, seed=0).video[0]
    engine.save_video(video, f"./test_outputs/{prompt}_cogvideo_pab_{num_gpus}.mp4")


@pytest.mark.parametrize("num_gpus", [1])
def test_low_mem(num_gpus):
    config = CogVideoXConfig(num_gpus=num_gpus, cpu_offload=True, vae_tiling=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt, seed=0).video[0]
    engine.save_video(video, f"./test_outputs/{prompt}_cogvideo_low_mem_{num_gpus}.mp4")
