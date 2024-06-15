import argparse
import os

os.environ["TOKENIZERS_PARALLELISM"] = "true"

import colossalai
import torch
import torch._dynamo.config
from colossalai.cluster import DistCoordinator

from opendit.core.parallel_mgr import set_parallel_manager
from opendit.core.skip_mgr import set_skip_manager
from opendit.models.opensora import IDDPM, STDiT2_XL_2, T5Encoder, VideoAutoencoderKL, save_sample, text_preprocessing
from opendit.utils.utils import set_seed, str_to_dtype


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

    set_skip_manager(
        steps=args.scheduler_num_sampling_steps,
        cross_skip=True,
        cross_threshold=700,
        cross_gap=5,
        spatial_skip=True,
        spatial_threshold=700,
        spatial_gap=3,
        temporal_skip=True,
        temporal_threshold=700,
        temporal_gap=5,
    )

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
        "Time Lapse of the rising sun over a tree in an open rural landscape, with clouds in the blue sky beautifully playing with the rays of light",
        "The vibrant beauty of a sunflower field. The sunflowers, with their bright yellow petals and dark brown centers, are in full bloom, creating a stunning contrast against the green leaves and stems. The sunflowers are arranged in neat rows, creating a sense of order and symmetry. The sun is shining brightly, casting a warm glow on the flowers and highlighting their intricate details. The video is shot from a low angle, looking up at the sunflowers, which adds a sense of grandeur and awe to the scene. The sunflowers are the main focus of the video, with no other objects or people present. The video is a celebration of nature's beauty and the simple joy of a sunny day in the countryside.",
        "Snow falling over multiple houses and trees on winter landscape against night sky. christmas festivity and celebration concept.",
        "A vibrant underwater scene. A group of blue fish, with yellow fins, are swimming around a coral reef. The coral reef is a mix of brown and green, providing a natural habitat for the fish. The water is a deep blue, indicating a depth of around 30 feet. The fish are swimming in a circular pattern around the coral reef, indicating a sense of motion and activity. The overall scene is a beautiful representation of marine life.",
        "A bustling city street at night, filled with the glow of car headlights and the ambient light of streetlights. The scene is a blur of motion, with cars speeding by and pedestrians navigating the crosswalks. The cityscape is a mix of towering buildings and illuminated signs, creating a vibrant and dynamic atmosphere. The perspective of the video is from a high angle, providing a bird's eye view of the street and its surroundings. The overall style of the video is dynamic and energetic, capturing the essence of urban life at night.",
        "A snowy forest landscape with a dirt road running through it. The road is flanked by trees covered in snow, and the ground is also covered in snow. The sun is shining, creating a bright and serene atmosphere. The road appears to be empty, and there are no people or animals visible in the video. The style of the video is a natural landscape shot, with a focus on the beauty of the snowy forest and the peacefulness of the road.",
        "A serene night scene in a forested area. The first frame shows a tranquil lake reflecting the star-filled sky above. The second frame reveals a beautiful sunset, casting a warm glow over the landscape. The third frame showcases the night sky, filled with stars and a vibrant Milky Way galaxy. The video is a time-lapse, capturing the transition from day to night, with the lake and forest serving as a constant backdrop. The style of the video is naturalistic, emphasizing the beauty of the night sky and the peacefulness of the forest.",
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
    # torch.compiler.reset()
    # torch._dynamo.config.accumulated_cache_size_limit = 256
    # model = torch.compile(model)
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

            # Skip if the sample already exists
            # This is useful for resuming sampling VBench

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
    main(args)
