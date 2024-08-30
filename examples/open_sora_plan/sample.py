from videosys import OpenSoraPlanConfig, VideoSysEngine


def run_base():
    # num frames: 65 or 221
    # change num_gpus for multi-gpu inference
    config = OpenSoraPlanConfig(num_frames=65, num_gpus=1)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(
        prompt=prompt,
        guidance_scale=7.5,
        num_inference_steps=150,
    ).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


def run_pab():
    config = OpenSoraPlanConfig(num_gpus=1, enable_pab=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


if __name__ == "__main__":
    run_base()
    # run_pab()
