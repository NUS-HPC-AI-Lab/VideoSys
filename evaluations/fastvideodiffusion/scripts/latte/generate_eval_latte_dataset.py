# Adapted from Latte

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Latte: https://github.com/Vchitect/Latte
# --------------------------------------------------------

import argparse
import os

import imageio
import torch
from diffusers.models import AutoencoderKL, AutoencoderKLTemporalDecoder
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.schedulers.scheduling_dpmsolver_singlestep import DPMSolverSinglestepScheduler
from omegaconf import OmegaConf
from torchvision.utils import save_image
from transformers import T5EncoderModel, T5Tokenizer

from opendit.core.skip_mgr import set_skip_manager
from opendit.models.latte import LattePipeline, LatteT2V
from opendit.utils.utils import merge_args
from evaluations.fastvideodiffusion.eval.utils import load_eval_prompts

def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    set_skip_manager(
        steps=args.num_sampling_steps,
        cross_skip=args.cross_skip,
        cross_threshold=args.cross_threshold,
        cross_gap=args.cross_gap,
        spatial_skip=args.spatial_skip,
        spatial_threshold=args.spatial_threshold,
        spatial_gap=args.spatial_gap,
        temporal_skip=args.temporal_skip,
        temporal_threshold=args.temporal_threshold,
        temporal_gap=args.temporal_gap,
    )

    transformer_model = LatteT2V.from_pretrained(
        args.pretrained_model_path, subfolder="transformer", video_length=args.video_length
    ).to(device, dtype=torch.float16)

    if args.enable_vae_temporal_decoder:
        vae = AutoencoderKLTemporalDecoder.from_pretrained(
            args.pretrained_model_path, subfolder="vae_temporal_decoder", torch_dtype=torch.float16
        ).to(device)
    else:
        vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae", torch_dtype=torch.float16).to(
            device
        )
    tokenizer = T5Tokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        args.pretrained_model_path, subfolder="text_encoder", torch_dtype=torch.float16
    ).to(device)

    # set eval mode
    transformer_model.eval()
    vae.eval()
    text_encoder.eval()

    if args.sample_method == "DDIM":
        scheduler = DDIMScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
            clip_sample=False,
        )
    elif args.sample_method == "EulerDiscrete":
        scheduler = EulerDiscreteScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "DDPM":
        scheduler = DDPMScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
            clip_sample=False,
        )
    elif args.sample_method == "DPMSolverMultistep":
        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "DPMSolverSinglestep":
        scheduler = DPMSolverSinglestepScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "PNDM":
        scheduler = PNDMScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "HeunDiscrete":
        scheduler = HeunDiscreteScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "EulerAncestralDiscrete":
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "DEISMultistep":
        scheduler = DEISMultistepScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )
    elif args.sample_method == "KDPM2AncestralDiscrete":
        scheduler = KDPM2AncestralDiscreteScheduler.from_pretrained(
            args.pretrained_model_path,
            subfolder="scheduler",
            beta_start=args.beta_start,
            beta_end=args.beta_end,
            beta_schedule=args.beta_schedule,
            variance_type=args.variance_type,
        )

    videogen_pipeline = LattePipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, scheduler=scheduler, transformer=transformer_model
    ).to(device)

    if not os.path.exists(args.save_img_path):
        os.makedirs(args.save_img_path)
    eval_prompts_dict = load_eval_prompts(args.eval_dataset)
    print('Generate eval datasets now!')
    print(f"Number of eval prompts: {len(eval_prompts_dict)}\n")
    # video_grids = []
    for num_prompt, (id, prompt) in enumerate(eval_prompts_dict.items()):
        print(f"Processing | prompt: ({prompt}) | id: ({id})")
        
        videos = videogen_pipeline(
            prompt,
            video_length=args.video_length,
            height=args.image_size[0],
            width=args.image_size[1],
            num_inference_steps=args.num_sampling_steps,
            guidance_scale=args.guidance_scale,
            enable_temporal_attentions=args.enable_temporal_attentions,
            num_images_per_prompt=1,
            mask_feature=True,
            enable_vae_temporal_decoder=args.enable_vae_temporal_decoder,
        ).video
        if videos.shape[1] == 1:
            save_image(videos[0][0], args.save_img_path + prompt.replace(" ", "_") + ".png")
        else:
            save_path = args.save_img_path + f'{id}' + ".mp4" # TODO edit save path
            imageio.mimwrite(
                save_path, videos[0], fps=8
            )
            print(f"Saved eval video to {save_path}!\n")

    print('Finish generating eval datasets now!')
    print(f"Number of eval videos: {len(eval_prompts_dict)}\n")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--save_img_path", type=str, default="./samples/latte/")
    parser.add_argument("--pretrained_model_path", type=str, default="maxin-cn/Latte-1")
    parser.add_argument("--model", type=str, default="LatteT2V")
    parser.add_argument("--video_length", type=int, default=16)
    parser.add_argument("--image_size", nargs="+")
    parser.add_argument("--beta_start", type=float, default=0.0001)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--beta_schedule", type=str, default="linear")
    parser.add_argument("--variance_type", type=str, default="learned_range")
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_time", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--sample_method", type=str, default="DDIM")
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--enable_temporal_attentions", action="store_true")
    parser.add_argument("--enable_vae_temporal_decoder", action="store_true")
    parser.add_argument("--text_prompt", nargs="+")
    # skip
    parser.add_argument("--spatial_skip", action="store_true", help="Enable spatial attention skip")
    parser.add_argument("--spatial_threshold", type=int, default=700, help="Spatial attention threshold")
    parser.add_argument("--spatial_gap", type=int, default=3, help="Spatial attention gap")
    parser.add_argument("--temporal_skip", action="store_true", help="Enable temporal attention skip")
    parser.add_argument("--temporal_threshold", type=int, default=700, help="Temporal attention threshold")
    parser.add_argument("--temporal_gap", type=int, default=5, help="Temporal attention gap")
    parser.add_argument("--cross_skip", action="store_true", help="Enable cross attention skip")
    parser.add_argument("--cross_threshold", type=int, default=700, help="Cross attention threshold")
    parser.add_argument("--cross_gap", type=int, default=5, help="Cross attention gap")
    # eval
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--eval_dataset", type=str, default="./evaluations/fastvideodiffusion/datasets/webvid_selected.csv")
    
    args = parser.parse_args()

    config_args = OmegaConf.load(args.config)
    args = merge_args(args, config_args)

    main(args)
