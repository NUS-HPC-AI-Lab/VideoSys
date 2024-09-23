from videosys import VchitectConfig, VideoSysEngine


def run_base():
    # change num_gpus for multi-gpu inference
    config = VchitectConfig("Vchitect/Vchitect-2.0-2B", num_gpus=1)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    # seed=-1 means random seed. >0 means fixed seed.
    # WxH: 480x288 624x352 432x240 768x432
    video = engine.generate(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=100,
        guidance_scale=7.5,
        width=480,
        height=288,
        frames=40,
        seed=0,
    ).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


def run_pab():
    config = VchitectConfig("Vchitect/Vchitect-2.0-2B", enable_pab=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


def run_low_mem():
    config = VchitectConfig("Vchitect/Vchitect-2.0-2B", cpu_offload=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


if __name__ == "__main__":
    run_base()
    # run_pab()
    # run_low_mem()
