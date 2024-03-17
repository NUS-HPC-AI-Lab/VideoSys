# Adapted from OpenSora

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# OpenSora: https://github.com/hpcaitech/Open-Sora
# --------------------------------------------------------


import argparse
import os

import torch

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


def main(args):
    # ======================================================
    # 1. args
    # ======================================================
    print(f"Args: {args}\n\n", end="")

    # ======================================================
    # 2. runtime variables
    # ======================================================
    torch.set_grad_enabled(False)
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = str_to_dtype(args.dtype)
    prompts = [
        "The majestic beauty of a waterfall cascading down a cliff into a serene lake. The waterfall, with its powerful flow, is the central focus of the video. The surrounding landscape is lush and green, with trees and foliage adding to the natural beauty of the scene. The camera angle provides a bird's eye view of the waterfall, allowing viewers to appreciate the full height and grandeur of the waterfall. The video is a stunning representation of nature's power and beauty.",
        "A majestic lion in its natural habitat. The lion, with its golden fur, is seen walking through a lush green field. The field is dotted with bushes and trees, providing a serene and natural backdrop. The lion's movement is captured in three frames, each showing the lion in a different position, giving a sense of its journey through the field. The overall style of the video is realistic and naturalistic, capturing the beauty of the lion in its environment.",
        "A lively scene in a park, where a large flock of pigeons is seen in the grassy field. The birds are scattered across the field, some standing, some walking, and some pecking at the ground. The park is lush with green grass and trees, providing a natural habitat for the birds. In the background, there are playground equipment and a building, indicating that the park is located in an urban area. The video is shot in daylight, with the sunlight casting a warm glow on the scene. The overall style of the video is a real-life, candid capture of a moment in the park, showcasing the interaction between the birds and their environment.",
    ]

    # ======================================================
    # 3. build model & load weights
    # ======================================================
    # 3.1. build model
    input_size = (args.num_frames, args.image_size[0], args.image_size[1])
    vae = VideoAutoencoderKL(args.vae_pretrained_path, split=8)
    latent_size = vae.get_latent_size(input_size)
    text_encoder = T5Encoder(args.text_pretrained_path, args.text_max_length, device=device)
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

    # 3.2. move to device & eval
    vae = vae.to(device, dtype).eval()
    model = model.to(device, dtype).eval()

    # 3.3. build scheduler
    scheduler = IDDPM(args.scheduler_num_sampling_steps, cfg_scale=args.scheduler_cfg_scale)

    # 3.4. support for multi-resolution
    model_args = dict()
    # if args.multi_resolution:
    #     image_size = args.image_size
    #     hw = torch.tensor([image_size], device=device, dtype=dtype).repeat(args.batch_size, 1)
    #     ar = torch.tensor([[image_size[0] / image_size[1]]], device=device, dtype=dtype).repeat(args.batch_size, 1)
    #     model_args["data_info"] = dict(ar=ar, hw=hw)

    # ======================================================
    # 4. inference
    # ======================================================
    sample_idx = 0
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    for i in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[i : i + args.batch_size]
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
            print(f"Prompt: {batch_prompts[idx]}")
            save_path = os.path.join(save_dir, f"sample_{sample_idx}")
            save_sample(sample, fps=args.fps, save_path=save_path)
            sample_idx += 1


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

    # scheduler
    parser.add_argument("--scheduler_num_sampling_steps", type=int, default=100)
    parser.add_argument("--scheduler_cfg_scale", type=int, default=7.0)

    # speedup
    parser.add_argument("--enable_layernorm_kernel", action="store_true", help="Enable apex layernorm kernel")
    parser.add_argument("--enable_flashattn", action="store_true", help="Enable flashattn kernel")

    args = parser.parse_args()
    main(args)
