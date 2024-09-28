import pytest

from videosys import VchitectConfig, VideoSysEngine


@pytest.mark.parametrize("num_gpus", [1, 2])
@pytest.mark.parametrize("model_path", ["Vchitect/Vchitect-2.0-2B", "Vchitect/Vchitect-2.0-5B"])
def test_base(num_gpus, model_path):
    config = VchitectConfig(model_path=model_path, num_gpus=num_gpus)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt, seed=0).video[0]
    engine.save_video(video, f"./test_outputs/{prompt}_vchitect_{model_path}_base_{num_gpus}.mp4")


@pytest.mark.parametrize("num_gpus", [1])
@pytest.mark.parametrize("model_path", ["Vchitect/Vchitect-2.0-2B"])
def test_pab(num_gpus, model_path):
    config = VchitectConfig(model_path=model_path, num_gpus=num_gpus, enable_pab=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt, seed=0).video[0]
    engine.save_video(video, f"./test_outputs/{prompt}_vchitect_{model_path}_pab_{num_gpus}.mp4")


@pytest.mark.parametrize("num_gpus", [1])
@pytest.mark.parametrize("model_path", ["Vchitect/Vchitect-2.0-2B"])
def test_low_mem(num_gpus, model_path):
    config = VchitectConfig(
        model_path=model_path,
        num_gpus=num_gpus,
        cpu_offload=True,
    )
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt, seed=0).video[0]
    engine.save_video(video, f"./test_outputs/{prompt}_vchitect_{model_path}_low_mem_{num_gpus}.mp4")
