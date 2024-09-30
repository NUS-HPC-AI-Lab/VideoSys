from videosys import OpenSoraConfig, VideoSysEngine, VideoSysRayEngine


def run_base():
    # change num_gpus for multi-gpu inference
    # sampling parameters are defined in the config
    config = OpenSoraConfig(num_sampling_steps=30, cfg_scale=7.0, num_gpus=1)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    # num frames: 2s, 4s, 8s, 16s
    # resolution: 144p, 240p, 360p, 480p, 720p
    # aspect ratio: 9:16, 16:9, 3:4, 4:3, 1:1
    # seed=-1 means random seed. >0 means fixed seed.
    video = engine.generate(
        prompt=prompt,
        resolution="480p",
        aspect_ratio="9:16",
        num_frames="2s",
        seed=114514,
    ).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


def run_low_mem():
    config = OpenSoraConfig(cpu_offload=True, tiling_size=1)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")


def run_pab():
    config = OpenSoraConfig(enable_pab=True)
    engine = VideoSysEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")

def run_dist():
    config = OpenSoraConfig(num_sampling_steps=30, cfg_scale=7.0, num_gpus=4, ray_address="auto")
    engine = VideoSysRayEngine(config)

    prompt = "Sunset over the sea."
    video = engine.generate(prompt).video[0]
    engine.save_video(video, f"./outputs/{prompt}.mp4")

if __name__ == "__main__":
    run_base()
    # run_dist()
    # run_low_mem()
    # run_pab()
