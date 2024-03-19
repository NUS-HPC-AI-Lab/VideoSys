# Adapted from OpenSora

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# OpenSora: https://github.com/hpcaitech/Open-Sora
# --------------------------------------------------------


import argparse
import os

import colossalai
import torch
import torch.distributed as dist
from colossalai.utils import get_current_device

from opendit.core.parallel_mgr import get_data_parallel_rank, get_sequence_parallel_rank, set_parallel_manager
from opendit.embed.t5_text_emb import T5Encoder
from opendit.models.opensora.scheduler import IDDPM
from opendit.models.opensora.stdit import STDiT_XL_2
from opendit.utils.utils import set_seed, str_to_dtype
from opendit.vae.reconstruct import save_sample
from opendit.vae.wrapper import VideoAutoencoderKL

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_prompts(prompt_path):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts


@torch.no_grad()
def main(args):
    # ======================================================
    # 1. Args
    # ======================================================
    print(f"Args: {args}\n\n", end="")

    # ==============================
    # 2. Initialize Distributed Training
    # ==============================
    colossalai.launch_from_torch({}, seed=args.seed)
    device = get_current_device()
    dtype = str_to_dtype(args.dtype)

    # ==============================
    # 3. Initialize Process Group
    # ==============================
    sp_size = args.sequence_parallel_size
    dp_size = dist.get_world_size() // sp_size
    set_parallel_manager(dp_size, sp_size, dp_axis=0, sp_axis=1)
    dp_rank = get_data_parallel_rank()
    sp_rank = get_sequence_parallel_rank()

    # ======================================================
    # 4. Runtime variables
    # ======================================================
    set_seed(args.seed)
    prompts = [
        "The majestic beauty of a waterfall cascading down a cliff into a serene lake. The waterfall, with its powerful flow, is the central focus of the video. The surrounding landscape is lush and green, with trees and foliage adding to the natural beauty of the scene. The camera angle provides a bird's eye view of the waterfall, allowing viewers to appreciate the full height and grandeur of the waterfall. The video is a stunning representation of nature's power and beauty.",
        "A soaring drone footage captures the majestic beauty of a coastal cliff, its red and yellow stratified rock faces rich in color and against the vibrant turquoise of the sea. Seabirds can be seen taking flight around the cliff's precipices. As the drone slowly moves from different angles, the changing sunlight casts shifting shadows that highlight the rugged textures of the cliff and the surrounding calm sea. The water gently laps at the rock base and the greenery that clings to the top of the cliff, and the scene gives a sense of peaceful isolation at the fringes of the ocean. The video captures the essence of pristine natural beauty untouched by human structures.",
        "A serene night scene in a forested area. The first frame shows a tranquil lake reflecting the star-filled sky above. The second frame reveals a beautiful sunset, casting a warm glow over the landscape. The third frame showcases the night sky, filled with stars and a vibrant Milky Way galaxy. The video is a time-lapse, capturing the transition from day to night, with the lake and forest serving as a constant backdrop. The style of the video is naturalistic, emphasizing the beauty of the night sky and the peacefulness of the forest.",
        "A bustling city street at night, filled with the glow of car headlights and the ambient light of streetlights. The scene is a blur of motion, with cars speeding by and pedestrians navigating the crosswalks. The cityscape is a mix of towering buildings and illuminated signs, creating a vibrant and dynamic atmosphere. The perspective of the video is from a high angle, providing a bird's eye view of the street and its surroundings. The overall style of the video is dynamic and energetic, capturing the essence of urban life at night.",
    ]

    # ======================================================
    # 5. Build model & load weights
    # ======================================================
    # 5.1 build model
    input_size = (args.num_frames, args.image_size[0], args.image_size[1])
    vae = VideoAutoencoderKL(args.vae_pretrained_path, split=8)
    latent_size = vae.get_latent_size(input_size)
    text_encoder = T5Encoder(
        args.text_pretrained_path,
        args.text_max_length,
        shardformer=args.text_speedup,
        device=device,
    )
    model = STDiT_XL_2(
        from_pretrained=args.model_pretrained_path,
        time_scale=args.model_time_scale,
        space_scale=args.model_space_scale,
        input_size=latent_size,
        in_channels=vae.out_channels,
        caption_channels=text_encoder.output_dim,
        model_max_length=text_encoder.model_max_length,
        dtype=dtype,
        enable_flashattn=args.enable_flashattn,
        enable_layernorm_kernel=args.enable_layernorm_kernel,
    )
    text_encoder.y_embedder = model.y_embedder  # hack for classifier-free guidance

    # 5.2 move to device & eval
    vae = vae.to(device, dtype).eval()
    model = model.to(device, dtype).eval()
    model_args = dict()

    # 5.3 build scheduler
    scheduler = IDDPM(args.scheduler_num_sampling_steps, cfg_scale=args.scheduler_cfg_scale)

    # ======================================================
    # 6. inference
    # ======================================================
    # makedir
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    # config data parallel for sample
    if dp_size > 1:
        assert len(prompts) % dp_size == 0, (
            "For data parallel, sample number should be divided by dp size, ",
            f"but got sample number {len(prompts)} and dp size {dp_size}",
        )
    local_prompt_len = len(prompts) // dp_size
    start_idx = local_prompt_len * dp_rank
    sample_idx = start_idx
    dist.barrier()

    for i in range(0, local_prompt_len, args.batch_size):
        batch_prompts = prompts[start_idx + i : start_idx + i + args.batch_size]
        samples = scheduler.sample(
            model,
            text_encoder,
            z_size=(vae.out_channels, *latent_size),
            prompts=batch_prompts,
            device=device,
            additional_args=model_args,
        )
        samples = vae.decode(samples.to(dtype))
        for idx, sample in enumerate(samples):
            if sp_rank == 0:
                print(f"Prompt: {batch_prompts[idx]}")
                save_path = os.path.join(save_dir, f"sample_{sample_idx}")
                save_sample(sample, fps=args.fps, save_path=save_path)
            sample_idx += 1
        dist.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # sample
    parser.add_argument("--num_frames", type=int, default=64)
    parser.add_argument("--fps", type=int, default=24 // 2)
    parser.add_argument("--image_size", nargs="+", type=int, default=[512, 512])
    parser.add_argument("--save_dir", type=str, default="./samples")

    # runtime
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="fp32")

    # model
    parser.add_argument("--model_space_scale", type=float, default=1.0)
    parser.add_argument("--model_time_scale", type=float, default=1.0)
    parser.add_argument("--model_pretrained_path", type=str, required=True)

    # vae
    parser.add_argument("--vae_pretrained_path", type=str, default="stabilityai/sd-vae-ft-ema")

    # text encoer
    parser.add_argument("--text_pretrained_path", type=str, default="t5-v1_1-xxl")
    parser.add_argument("--text_max_length", type=int, default=120)
    parser.add_argument("--text_speedup", action="store_true")

    # scheduler
    parser.add_argument("--scheduler_num_sampling_steps", type=int, default=100)
    parser.add_argument("--scheduler_cfg_scale", type=int, default=7.0)

    # speedup
    parser.add_argument("--enable_layernorm_kernel", action="store_true", help="Enable apex layernorm kernel")
    parser.add_argument("--enable_flashattn", action="store_true", help="Enable flashattn kernel")

    # parallel
    parser.add_argument("--sequence_parallel_size", type=int, default=1, help="Sequence parallel size, enable if > 1")

    args = parser.parse_args()
    main(args)
