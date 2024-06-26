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

from opendit.models.latte import LattePipeline, LatteT2V


def main(args):
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # video_grids = []
    for num_prompt, prompt in enumerate(args.text_prompt):
        print("Processing the ({}) prompt".format(prompt))
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
            imageio.mimwrite(
                args.save_img_path + prompt.replace(" ", "_") + "_%04d" % args.run_time + ".mp4", videos[0], fps=8
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    main(OmegaConf.load(args.config))
