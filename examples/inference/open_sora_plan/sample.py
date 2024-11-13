from videosys import OpenSoraPlanConfig, VideoSysEngine


def run_base():
    # open-sora-plan v1.2.0
    # transformer_type (len, res): 93x480p 93x720p 29x480p 29x720p
    # change num_gpus for multi-gpu inference
    config = OpenSoraPlanConfig(version="v120", transformer_type="29x480p", num_gpus=1)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    # seed=-1 means random seed. >0 means fixed seed.
    video = engine.generate(
        prompt=prompt,
        guidance_scale=7.5,
        num_inference_steps=100,
        seed=-1,
    ).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


def run_low_mem():
    config = OpenSoraPlanConfig(cpu_offload=True, enable_tiling=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


def run_pab():
    config = OpenSoraPlanConfig(enable_pab=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


def run_v110():
    # open-sora-plan v1.1.0
    # transformer_type: 65x512x512 or 221x512x512
    # change num_gpus for multi-gpu inference
    config = OpenSoraPlanConfig(version="v110", transformer_type="65x512x512", num_gpus=1)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    # seed=-1 means random seed. >0 means fixed seed.
    video = engine.generate(
        prompt=prompt,
        guidance_scale=7.5,
        num_inference_steps=150,
        seed=-1,
    ).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


if __name__ == "__main__":
    run_base()
    # run_low_mem()
    # run_pab()
    # run_v110()
