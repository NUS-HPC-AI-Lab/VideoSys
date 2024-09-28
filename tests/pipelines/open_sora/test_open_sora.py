import pytest

from videosys import OpenSoraConfig, VideoSysEngine
from videosys.utils.test import empty_cache


@pytest.mark.parametrize("num_gpus", [1, 2])
@empty_cache
def test_base(num_gpus):
    config = OpenSoraConfig(num_gpus=num_gpus)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt, seed=0).video[0]
    engine.save_video(video, f"./test_outputs/{prompt}_open_sora_{num_gpus}.mp4")


@pytest.mark.parametrize("num_gpus", [1])
@empty_cache
def test_pab(num_gpus):
    config = OpenSoraConfig(num_gpus=num_gpus, enable_pab=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt, seed=0).video[0]
    engine.save_video(video, f"./test_outputs/{prompt}_open_sora_pab_{num_gpus}.mp4")


@pytest.mark.parametrize("num_gpus", [1])
@empty_cache
def test_low_mem(num_gpus):
    config = OpenSoraConfig(num_gpus=num_gpus, cpu_offload=True, tiling_size=1)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt, seed=0).video[0]
    engine.save_video(video, f"./test_outputs/{prompt}_open_sora_low_mem_{num_gpus}.mp4")
