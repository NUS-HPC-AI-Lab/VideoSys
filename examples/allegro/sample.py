from videosys import AllegroConfig, AllegroPABConfig, VideoSysEngine
import io
import imageio
import torch
def run_base():
    # num frames: 65 or 221
    # change num_gpus for multi-gpu inference
    config = AllegroConfig(model_path="rhymes-ai/Allegro",
            cpu_offload=False,
            num_gpus=4)
    engine = VideoSysEngine(config)

    positive_prompt = """
(masterpiece), (best quality), (ultra-detailed), (unwatermarked), 
{} 
emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, 
sharp focus, high budget, cinemascope, moody, epic, gorgeous
"""

    negative_prompt = """
nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
"""

    user_prompt = "A seaside harbor with bright sunlight and sparkling seawater, with many boats in the water. From an aerial view, the boats vary in size and color, some moving and some stationary. Fishing boats in the water suggest that this location might be a popular spot for docking fishing boats."
    num_step, cfg_scale, rand_seed = 100, 7.5, 42
    input_prompt = positive_prompt.format(user_prompt.lower().strip())


    video = engine.generate(
        input_prompt, 
        negative_prompt=negative_prompt, 
        num_frames=88,
        height=720,
        width=1280,
        num_inference_steps=num_step,
        guidance_scale=cfg_scale,
        max_sequence_length=512,
        seed=rand_seed
    ).video[0]

    engine.save_video(video, f"./outputs/{user_prompt}.mp4")


def run_low_mem():

    # change num_gpus for multi-gpu inference
    config = AllegroConfig(model_path="rhymes-ai/Allegro",
            cpu_offload=True)
    engine = VideoSysEngine(config)


    positive_prompt = """
(masterpiece), (best quality), (ultra-detailed), (unwatermarked), 
{} 
emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, 
sharp focus, high budget, cinemascope, moody, epic, gorgeous
"""

    negative_prompt = """
nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
"""


    user_prompt = "A seaside harbor with bright sunlight and sparkling seawater, with many boats in the water. From an aerial view, the boats vary in size and color, some moving and some stationary. Fishing boats in the water suggest that this location might be a popular spot for docking fishing boats."
    num_step, cfg_scale, rand_seed = 100, 7.5, 42
    input_prompt = positive_prompt.format(user_prompt.lower().strip())


    video = engine.generate(
        input_prompt, 
        negative_prompt=negative_prompt, 
        num_frames=88,
        height=720,
        width=1280,
        num_inference_steps=num_step,
        guidance_scale=cfg_scale,
        max_sequence_length=512,
        seed=rand_seed
    ).video[0]
    engine.save_video(video, f"./outputs/{user_prompt}.mp4")


def run_pab():

    config = AllegroConfig(model_path="rhymes-ai/Allegro",
                    cpu_offload=False, enable_tiling=True, num_gpus=4, enable_pab=True)
    engine = VideoSysEngine(config)

    positive_prompt = """
(masterpiece), (best quality), (ultra-detailed), (unwatermarked), 
{} 
emotional, harmonious, vignette, 4k epic detailed, shot on kodak, 35mm photo, 
sharp focus, high budget, cinemascope, moody, epic, gorgeous
"""

    negative_prompt = """
nsfw, lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, 
low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry.
"""


    user_prompt = "A seaside harbor with bright sunlight and sparkling seawater, with many boats in the water. From an aerial view, the boats vary in size and color, some moving and some stationary. Fishing boats in the water suggest that this location might be a popular spot for docking fishing boats."
    num_step, cfg_scale, rand_seed = 100, 7.5, 42
    input_prompt = positive_prompt.format(user_prompt.lower().strip())


    video = engine.generate(
        input_prompt, 
        negative_prompt=negative_prompt, 
        num_frames=88,
        height=720,
        width=1280,
        num_inference_steps=num_step,
        guidance_scale=cfg_scale,
        max_sequence_length=512,
        seed=rand_seed
    ).video[0]
    engine.save_video(video, f"./outputs/{user_prompt}.mp4")

if __name__ == "__main__":
    run_base()
    # run_pab()
    # run_low_mem()
