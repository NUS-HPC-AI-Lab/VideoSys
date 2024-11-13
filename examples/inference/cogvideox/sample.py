from videosys import CogVideoXConfig, VideoSysEngine


def run_base():
    # models: "THUDM/CogVideoX-2b" or "THUDM/CogVideoX-5b"
    # change num_gpus for multi-gpu inference
    config = CogVideoXConfig("THUDM/CogVideoX-2b", num_gpus=1)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    # num frames should be <= 49. resolution is fixed to 720p.
    # seed=-1 means random seed. >0 means fixed seed.
    video = engine.generate(
        prompt=prompt,
        guidance_scale=6,
        num_inference_steps=50,
        num_frames=49,
        seed=-1,
    ).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


def run_pab():
    config = CogVideoXConfig("THUDM/CogVideoX-2b", enable_pab=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


def run_low_mem():
    config = CogVideoXConfig("THUDM/CogVideoX-2b", cpu_offload=True, vae_tiling=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


if __name__ == "__main__":
    run_base()
    # run_pab()
    # run_low_mem()
