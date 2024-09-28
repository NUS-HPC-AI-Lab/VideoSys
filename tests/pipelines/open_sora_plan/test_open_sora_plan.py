import pytest

from videosys import OpenSoraPlanConfig, VideoSysEngine
from videosys.utils.test import empty_cache


@pytest.mark.parametrize("num_gpus", [1, 2])
@pytest.mark.parametrize("model", [("v120", "29x480p")])
@empty_cache
def test_base(num_gpus, model):
    config = OpenSoraPlanConfig(version=model[0], transformer_type=model[1], num_gpus=num_gpus)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt, seed=0).video[0]
    engine.save_video(video, f"./test_outputs/{prompt}_open_sora_plan_{model[0]}_{model[1]}_{num_gpus}.mp4")


@pytest.mark.parametrize("num_gpus", [1])
@pytest.mark.parametrize("model", [("v120", "29x480p")])
@empty_cache
def test_pab(num_gpus, model):
    config = OpenSoraPlanConfig(version=model[0], transformer_type=model[1], num_gpus=num_gpus, enable_pab=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt, seed=0).video[0]
    engine.save_video(video, f"./test_outputs/{prompt}_open_sora_plan_{model[0]}_{model[1]}_pab_{num_gpus}.mp4")


@pytest.mark.parametrize("num_gpus", [1])
@pytest.mark.parametrize("model", [("v120", "29x480p")])
@empty_cache
def test_low_mem(num_gpus, model):
    config = OpenSoraPlanConfig(
        version=model[0], transformer_type=model[1], num_gpus=num_gpus, cpu_offload=True, enable_tiling=True
    )
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt, seed=0).video[0]
    engine.save_video(video, f"./test_outputs/{prompt}_open_sora_plan_{model[0]}_{model[1]}_low_mem_{num_gpus}.mp4")
