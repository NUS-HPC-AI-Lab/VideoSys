import argparse
import os

import colossalai
import torch
from colossalai.cluster import DistCoordinator
from omegaconf import OmegaConf

from opendit.core.parallel_mgr import set_parallel_manager
from opendit.models.opensora import IDDPM, STDiT2_XL_2, T5Encoder, VideoAutoencoderKL, save_sample, text_preprocessing
from opendit.utils.utils import merge_args, set_seed, str_to_dtype


def main(args):
    # ======================================================
    # 1. cfg and init distributed env
    # ======================================================
    print(args)

    # init distributed
    if os.environ.get("WORLD_SIZE", None):
        use_dist = True
        colossalai.launch_from_torch({})
        coordinator = DistCoordinator()

        if coordinator.world_size > 1:
            set_parallel_manager(1, coordinator.world_size, dp_axis=0, sp_axis=1)
            enable_sequence_parallelism = True
        else:
            enable_sequence_parallelism = False
    else:
        use_dist = False
        enable_sequence_parallelism = False

    # ======================================================
    # 2. runtime variables
    # ======================================================
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = str_to_dtype(args.dtype)
    set_seed(seed=args.seed)
    prompts = [
        "Drone view of waves crashing against the rugged cliffs along Big Sur’s garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff’s edge. The steep drop from the road down to the beach is a dramatic feat, with the cliff’s edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway.",
        "A soaring drone footage captures the majestic beauty of a coastal cliff, its red and yellow stratified rock faces rich in color and against the vibrant turquoise of the sea. Seabirds can be seen taking flight around the cliff's precipices. As the drone slowly moves from different angles, the changing sunlight casts shifting shadows that highlight the rugged textures of the cliff and the surrounding calm sea. The water gently laps at the rock base and the greenery that clings to the top of the cliff, and the scene gives a sense of peaceful isolation at the fringes of the ocean. The video captures the essence of pristine natural beauty untouched by human structures.",
        "A majestic beauty of a waterfall cascading down a cliff into a serene lake. The waterfall, with its powerful flow, is the central focus of the video. The surrounding landscape is lush and green, with trees and foliage adding to the natural beauty of the scene. The camera angle provides a bird's eye view of the waterfall, allowing viewers to appreciate the full height and grandeur of the waterfall. The video is a stunning representation of nature's power and beauty.",
        "A vibrant scene of a snowy mountain landscape. The sky is filled with a multitude of colorful hot air balloons, each floating at different heights, creating a dynamic and lively atmosphere. The balloons are scattered across the sky, some closer to the viewer, others further away, adding depth to the scene.  Below, the mountainous terrain is blanketed in a thick layer of snow, with a few patches of bare earth visible here and there. The snow-covered mountains provide a stark contrast to the colorful balloons, enhancing the visual appeal of the scene.  In the foreground, a few cars can be seen driving along a winding road that cuts through the mountains. The cars are small compared to the vastness of the landscape, emphasizing the grandeur of the surroundings.  The overall style of the video is a mix of adventure and tranquility, with the hot air balloons adding a touch of whimsy to the otherwise serene mountain landscape. The video is likely shot during the day, as the lighting is bright and even, casting soft shadows on the snow-covered mountains.",
        "The vibrant beauty of a sunflower field. The sunflowers, with their bright yellow petals and dark brown centers, are in full bloom, creating a stunning contrast against the green leaves and stems. The sunflowers are arranged in neat rows, creating a sense of order and symmetry. The sun is shining brightly, casting a warm glow on the flowers and highlighting their intricate details. The video is shot from a low angle, looking up at the sunflowers, which adds a sense of grandeur and awe to the scene. The sunflowers are the main focus of the video, with no other objects or people present. The video is a celebration of nature's beauty and the simple joy of a sunny day in the countryside.",
        "Time Lapse of the rising sun over a tree in an open rural landscape, with clouds in the blue sky beautifully playing with the rays of light",
        "Snow falling over multiple houses and trees on winter landscape against night sky. christmas festivity and celebration concept.",
    ]

    # ======================================================
    # 3. build model & load weights
    # ======================================================
    # 3.1. build model
    input_size = (args.num_frames, *args.image_size)
    vae = VideoAutoencoderKL(
        from_pretrained="stabilityai/sd-vae-ft-ema",
        split=4,
    )
    latent_size = vae.get_latent_size(input_size)
    text_encoder = T5Encoder(
        from_pretrained="DeepFloyd/t5-v1_1-xxl",
        model_max_length=200,
        device=device,
        shardformer=args.enable_t5_speedup,
    )

    model = STDiT2_XL_2(
        from_pretrained=args.model_pretrained_path,
        input_sq_size=512,
        qk_norm=True,
        enable_flash_attn=args.enable_flashattn,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        enable_sequence_parallelism=enable_sequence_parallelism,
    )
    text_encoder.y_embedder = model.y_embedder  # hack for classifier-free guidance

    # 3.2. move to device & eval
    vae = vae.to(device, dtype).eval()
    model = model.to(device, dtype).eval()

    # 3.3. build scheduler
    scheduler = IDDPM(
        num_sampling_steps=args.scheduler_num_sampling_steps,
        cfg_scale=args.scheduler_cfg_scale,
        cfg_channel=3,
    )

    # 3.4. support for multi-resolution
    model_args = dict()
    image_size = args.image_size
    height = torch.tensor([image_size[0]], device=device, dtype=dtype).repeat(args.batch_size)
    width = torch.tensor([image_size[1]], device=device, dtype=dtype).repeat(args.batch_size)
    num_frames = torch.tensor([args.num_frames], device=device, dtype=dtype).repeat(args.batch_size)
    ar = torch.tensor([image_size[0] / image_size[1]], device=device, dtype=dtype).repeat(args.batch_size)
    if args.num_frames == 1:
        args.fps = 120
    fps = torch.tensor([args.fps], device=device, dtype=dtype).repeat(args.batch_size)
    model_args["height"] = height
    model_args["width"] = width
    model_args["num_frames"] = num_frames
    model_args["ar"] = ar
    model_args["fps"] = fps

    # ======================================================
    # 4. inference
    # ======================================================
    sample_idx = 0
    sample_name = "sample"
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)

    # 4.1. batch generation
    for i in range(0, len(prompts), args.batch_size):
        # 4.2 sample in hidden space
        batch_prompts_raw = prompts[i : i + args.batch_size]
        batch_prompts = [text_preprocessing(prompt) for prompt in batch_prompts_raw]
        # handle the last batch
        if len(batch_prompts_raw) < args.batch_size and args.multi_resolution == "STDiT2":
            model_args["height"] = model_args["height"][: len(batch_prompts_raw)]
            model_args["width"] = model_args["width"][: len(batch_prompts_raw)]
            model_args["num_frames"] = model_args["num_frames"][: len(batch_prompts_raw)]
            model_args["ar"] = model_args["ar"][: len(batch_prompts_raw)]
            model_args["fps"] = model_args["fps"][: len(batch_prompts_raw)]

        # 4.3. diffusion sampling
        old_sample_idx = sample_idx
        # generate multiple samples for each prompt
        for k in range(args.num_samples):
            sample_idx = old_sample_idx

            # sampling
            z = torch.randn(len(batch_prompts), vae.out_channels, *latent_size, device=device, dtype=dtype)
            samples = scheduler.sample(
                model,
                text_encoder,
                z=z,
                prompts=batch_prompts,
                device=device,
                additional_args=model_args,
            )
            samples = vae.decode(samples.to(dtype))

            # 4.4. save samples
            if not use_dist or coordinator.is_master():
                for idx, sample in enumerate(samples):
                    print(f"Prompt: {batch_prompts_raw[idx]}")
                    sample_name_suffix = f"_{sample_idx}_{batch_prompts_raw[idx][:30]}"
                    save_path = os.path.join(save_dir, f"{sample_name}{sample_name_suffix}")
                    save_path = f"{save_path}-{k}"
                    save_sample(sample, fps=args.fps // args.frame_interval, save_path=save_path)
                    sample_idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)

    # sample
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--frame_interval", type=int, default=3)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--image_size", nargs="+", type=int, default=[240, 426])
    parser.add_argument("--num_samples", type=int, default=1)

    # runtime
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="./samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="bf16")

    # model
    parser.add_argument("--model_pretrained_path", type=str, default="hpcai-tech/OpenSora-STDiT-v2-stage3")

    # scheduler
    parser.add_argument("--scheduler_num_sampling_steps", type=int, default=100)
    parser.add_argument("--scheduler_cfg_scale", type=int, default=7.0)

    # speedup
    parser.add_argument("--enable_flashattn", action="store_true", help="Enable flashattn kernel")
    parser.add_argument("--enable_t5_speedup", action="store_true", help="Enable t5 speedup")

    args = parser.parse_args()
    config_args = OmegaConf.load(args.config)
    args = merge_args(args, config_args)

    main(args)
