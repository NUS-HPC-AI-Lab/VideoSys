from videosys.pipelines.flux.pipeline_flux_pab import FluxConfig, FluxPipeline, FluxPABConfig
import torch
import time 


def run_base():
    # change num_gpus for multi-gpu inference
    # sampling parameters are defined in the config
    
    # config = OpenSoraConfig(num_sampling_steps=30, cfg_scale=7.0, num_gpus=1)
    # engine = VideoSysEngine(config)
    config = FluxConfig()
    
    pipe = FluxPipeline(
        config = config
    )
    
    prompts = [
        "Sunset over the sea.",
        "A group of people wading around a large body of water.",
        "A cabinet towels a toilet and a sink.",
        "A few boys read comic books together outside.",
        "There is a man and a young girl snowboarding.",
        "A male snowboarder sits with his board on a snowy hill.",
        "A woman wearing a skirt shows off her tattoos.",
        "A large dog is tied up to a fire hydrant.",
        "A man making a pizza in a kitchen."
    ]

    # Generate and save images
    for prompt in prompts:
        start_time = time.time()
        image = pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cuda:0").manual_seed(0)
        ).images[0]
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"'{prompt}' | {elapsed_time:.2f} s.")
        pipe.save_image(image, f"./outputs/flux/{prompt}.png")

def run_pab():
    pab_config = FluxPABConfig(
        spatial_broadcast=True,
        spatial_threshold=[100, 930],
        spatial_range=5,
        temporal_broadcast=False,
        cross_broadcast=True,
        cross_threshold=[100, 930],
        cross_range=5,
        mlp_broadcast=True
    )
    config = FluxConfig(
        enable_pab=True,
        pab_config=pab_config)
    pipe = FluxPipeline(
        config = config
    )

    prompts = [
        "Sunset over the sea.",
        "A group of people wading around a large body of water.",
        "A cabinet towels a toilet and a sink.",
        "A few boys read comic books together outside.",
        "There is a man and a young girl snowboarding.",
        "A male snowboarder sits with his board on a snowy hill.",
        "A woman wearing a skirt shows off her tattoos.",
        "A large dog is tied up to a fire hydrant.",
        "A man making a pizza in a kitchen."
    ]

    for prompt in prompts:
        start_time = time.time()
        image = pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
            generator=torch.Generator("cuda:0").manual_seed(0)
        ).images[0]
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"'{prompt}' | {elapsed_time:.2f} s.")
        pipe.save_image(image, f"./outputs/flux-pab/{prompt.replace(' ', '_')}.png")

    # results = pipe(
    #     prompts,
    #     height=1024,
    #     width=1024,
    #     guidance_scale=3.5,
    #     num_inference_steps=50,
    #     max_sequence_length=512,
    #     generator=torch.Generator("cuda:0").manual_seed(0)
    # )

    # for idx, image in enumerate(results.images):
    #     safe_filename = f"./outputs/flux-pab-batch/image_{idx}.png"
    #     pipe.save_image(image, safe_filename)
        
        
if __name__ == "__main__":
    run_base()
    # run_low_mem()
    run_pab()
