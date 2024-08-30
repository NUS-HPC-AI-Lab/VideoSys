from videosys import LatteConfig, VideoSysEngine


def run_base():
    # change num_gpus for multi-gpu inference
    config = LatteConfig("maxin-cn/Latte-1", num_gpus=1)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    # video size is fixed to 16 frames, 512x512.
    video = engine.generate(
        prompt=prompt,
        guidance_scale=7.5,
        num_inference_steps=50,
    ).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


def run_pab():
    config = LatteConfig("maxin-cn/Latte-1", num_gpus=1)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


if __name__ == "__main__":
    run_base()
    # run_pab()
