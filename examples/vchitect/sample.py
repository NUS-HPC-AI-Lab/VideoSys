from videosys import VchitectXLConfig, VideoSysEngine


def run_base():
    # change num_gpus for multi-gpu inference
    config = VchitectXLConfig("Vchitect/Vchitect-2.0-2B", num_gpus=1)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    # seed=-1 means random seed. >0 means fixed seed.
    # HxW: 480x288  624x352 432x240 768x432
    video = engine.generate(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=100,
        guidance_scale=7.5,
        width=768,
        height=432,
        frames=40,
        seed=-1,
    ).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


if __name__ == "__main__":
    run_base()
