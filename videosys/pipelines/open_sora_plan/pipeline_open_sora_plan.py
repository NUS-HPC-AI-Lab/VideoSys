# Adapted from Open-Sora-Plan

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# Open-Sora-Plan: https://github.com/PKU-YuanGroup/Open-Sora-Plan
# --------------------------------------------------------

import html
import inspect
import math
import re
import urllib.parse as ul
from typing import Callable, List, Optional, Tuple, Union

import ftfy
import torch
import torch.distributed as dist
import tqdm
from bs4 import BeautifulSoup
from diffusers.models import AutoencoderKL, Transformer2DModel
from diffusers.schedulers import EulerAncestralDiscreteScheduler, PNDMScheduler
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoTokenizer, MT5EncoderModel, T5EncoderModel, T5Tokenizer

from videosys.core.pab.pab_mgr import PABConfig, set_pab_manager, update_steps
from videosys.core.pipeline.pipeline import VideoSysPipeline, VideoSysPipelineOutput
from videosys.models.autoencoders.autoencoder_kl_open_sora_plan_v110 import (
    CausalVAEModelWrapper as CausalVAEModelWrapperV110,
)
from videosys.models.autoencoders.autoencoder_kl_open_sora_plan_v120 import (
    CausalVAEModelWrapper as CausalVAEModelWrapperV120,
)
from videosys.models.transformers.open_sora_plan_v110_transformer_3d import LatteT2V
from videosys.models.transformers.open_sora_plan_v120_transformer_3d import OpenSoraT2V
from videosys.utils.logging import logger
from videosys.utils.utils import save_video, set_seed


class OpenSoraPlanV110PABConfig(PABConfig):
    def __init__(
        self,
        spatial_broadcast: bool = True,
        spatial_threshold: list = [100, 850],
        spatial_range: int = 2,
        temporal_broadcast: bool = True,
        temporal_threshold: list = [100, 850],
        temporal_range: int = 4,
        cross_broadcast: bool = True,
        cross_threshold: list = [100, 850],
        cross_range: int = 6,
        mlp_broadcast: bool = True,
        mlp_spatial_broadcast_config: dict = {
            738: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            714: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            690: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            666: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            642: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            618: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            594: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            570: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            546: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            522: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            498: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            474: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            450: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            426: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
        },
        mlp_temporal_broadcast_config: dict = {
            738: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            714: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            690: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            666: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            642: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            618: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            594: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            570: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            546: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            522: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            498: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            474: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            450: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
            426: {"block": [0, 1, 2, 3, 4, 5, 6], "skip_count": 2},
        },
    ):
        super().__init__(
            spatial_broadcast=spatial_broadcast,
            spatial_threshold=spatial_threshold,
            spatial_range=spatial_range,
            temporal_broadcast=temporal_broadcast,
            temporal_threshold=temporal_threshold,
            temporal_range=temporal_range,
            cross_broadcast=cross_broadcast,
            cross_threshold=cross_threshold,
            cross_range=cross_range,
            mlp_broadcast=mlp_broadcast,
            mlp_spatial_broadcast_config=mlp_spatial_broadcast_config,
            mlp_temporal_broadcast_config=mlp_temporal_broadcast_config,
        )


class OpenSoraPlanV120PABConfig(PABConfig):
    def __init__(
        self,
        spatial_broadcast: bool = True,
        spatial_threshold: list = [100, 850],
        spatial_range: int = 2,
        cross_broadcast: bool = True,
        cross_threshold: list = [100, 850],
        cross_range: int = 6,
    ):
        super().__init__(
            spatial_broadcast=spatial_broadcast,
            spatial_threshold=spatial_threshold,
            spatial_range=spatial_range,
            cross_broadcast=cross_broadcast,
            cross_threshold=cross_threshold,
            cross_range=cross_range,
        )


class OpenSoraPlanConfig:
    """
    This config is to instantiate a `OpenSoraPlanPipeline` class for video generation.

    To be specific, this config will be passed to engine by `VideoSysEngine(config)`.
    In the engine, it will be used to instantiate the corresponding pipeline class.
    And the engine will call the `generate` function of the pipeline to generate the video.
    If you want to explore the detail of generation, please refer to the pipeline class below.

    Args:
        transformer (str):
            The transformer model to use. Defaults to "LanguageBind/Open-Sora-Plan-v1.1.0".
        ae (str):
            The Autoencoder model to use. Defaults to "CausalVAEModel_4x8x8".
        text_encoder (str):
            The text encoder model to use. Defaults to "DeepFloyd/t5-v1_1-xxl".
        num_frames (int):
            The number of frames to generate. Must be one of [65, 221].
        num_gpus (int):
            The number of GPUs to use. Defaults to 1.
        enable_tiling (bool):
            Whether to enable tiling. Defaults to True.
        tile_overlap_factor (float):
            The overlap factor for tiling. Defaults to 0.25.
        enable_pab (bool):
            Whether to enable Pyramid Attention Broadcast. Defaults to False.
        pab_config (CogVideoXPABConfig):
            The configuration for Pyramid Attention Broadcast. Defaults to `LattePABConfig()`.

    Examples:
        ```python
        from videosys import OpenSoraPlanConfig, VideoSysEngine

        # num frames: 65 or 221
        # change num_gpus for multi-gpu inference
        config = OpenSoraPlanConfig(num_frames=65, num_gpus=1)
        engine = VideoSysEngine(config)

        prompt = "Sunset over the sea."
        video = engine.generate(
            prompt=prompt,
            guidance_scale=7.5,
            num_inference_steps=150,
        ).video[0]
        engine.save_video(video, f"./outputs/{prompt}.mp4")
        ```
    """

    def __init__(
        self,
        version: str = "v120",
        transformer_type: str = "29x480p",
        transformer: str = None,
        text_encoder: str = None,
        # ======= distributed ========
        num_gpus: int = 1,
        # ======= memory =======
        cpu_offload: bool = False,
        enable_tiling: bool = True,
        tile_overlap_factor: float = 0.25,
        # ======= pab ========
        enable_pab: bool = False,
        pab_config: PABConfig = None,
    ):
        self.pipeline_cls = OpenSoraPlanPipeline

        # get version
        assert version in ["v110", "v120"], f"Unknown Open-Sora-Plan version: {version}"
        self.version = version
        self.transformer_type = transformer_type

        # check transformer_type
        if version == "v110":
            assert transformer_type in ["65x512x512", "221x512x512"]
        elif version == "v120":
            assert transformer_type in ["93x480p", "93x720p", "29x480p", "29x720p"]
        self.num_frames = int(transformer_type.split("x")[0])

        # set default values according to version
        if version == "v110":
            transformer_default = "LanguageBind/Open-Sora-Plan-v1.1.0"
            text_encoder_default = "DeepFloyd/t5-v1_1-xxl"
        elif version == "v120":
            transformer_default = "LanguageBind/Open-Sora-Plan-v1.2.0"
            text_encoder_default = "google/mt5-xxl"
        self.text_encoder = text_encoder or text_encoder_default
        self.transformer = transformer or transformer_default

        # ======= distributed ========
        self.num_gpus = num_gpus
        # ======= memory ========
        self.cpu_offload = cpu_offload
        self.enable_tiling = enable_tiling
        self.tile_overlap_factor = tile_overlap_factor
        # ======= pab ========
        self.enable_pab = enable_pab
        if self.enable_pab and pab_config is None:
            if version == "v110":
                self.pab_config = OpenSoraPlanV110PABConfig()
            elif version == "v120":
                self.pab_config = OpenSoraPlanV120PABConfig()
        else:
            self.pab_config = pab_config


class OpenSoraPlanPipeline(VideoSysPipeline):
    r"""
    Pipeline for text-to-image generation using PixArt-Alpha.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. PixArt-Alpha uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel), specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`Transformer2DModel`]):
            A text conditioned `Transformer2DModel` to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
    """
    bad_punct_regex = re.compile(
        r"[" + "#®•©™&@·º½¾¿¡§~" + "\)" + "\(" + "\]" + "\[" + "\}" + "\{" + "\|" + "\\" + "\/" + "\*" + r"]{1,}"
    )  # noqa

    _optional_components = ["tokenizer", "text_encoder", "vae", "transformer", "scheduler"]
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(
        self,
        config: OpenSoraPlanConfig,
        tokenizer: Optional[T5Tokenizer] = None,
        text_encoder: Optional[T5EncoderModel] = None,
        vae: Optional[AutoencoderKL] = None,
        transformer: Optional[Transformer2DModel] = None,
        scheduler: Optional[PNDMScheduler] = None,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self._config = config

        # init
        if tokenizer is None:
            if config.version == "v110":
                tokenizer = T5Tokenizer.from_pretrained(config.text_encoder)
            elif config.version == "v120":
                tokenizer = AutoTokenizer.from_pretrained(config.text_encoder)

        if text_encoder is None:
            if config.version == "v110":
                text_encoder = T5EncoderModel.from_pretrained(config.text_encoder, torch_dtype=dtype)
            elif config.version == "v120":
                text_encoder = MT5EncoderModel.from_pretrained(
                    config.text_encoder, low_cpu_mem_usage=True, torch_dtype=dtype
                )

        if vae is None:
            if config.version == "v110":
                vae = CausalVAEModelWrapperV110(config.transformer, subfolder="vae").to(dtype=dtype)
            elif config.version == "v120":
                vae = CausalVAEModelWrapperV120(config.transformer, subfolder="vae").to(dtype=dtype)

        if transformer is None:
            if config.version == "v110":
                transformer = LatteT2V.from_pretrained(
                    config.transformer, subfolder=config.transformer_type, torch_dtype=dtype
                )
            elif config.version == "v120":
                transformer = OpenSoraT2V.from_pretrained(
                    config.transformer, subfolder=config.transformer_type, torch_dtype=dtype
                )

        if scheduler is None:
            if config.version == "v110":
                scheduler = PNDMScheduler()
            elif config.version == "v120":
                scheduler = EulerAncestralDiscreteScheduler()

        # setting
        if config.enable_tiling:
            vae.vae.enable_tiling()
            vae.vae.tile_overlap_factor = config.tile_overlap_factor
            vae.vae.tile_sample_min_size = 512
            vae.vae.tile_latent_min_size = 64
            vae.vae.tile_sample_min_size_t = 29
            vae.vae.tile_latent_min_size_t = 8
            # if low_mem:
            #     vae.vae.tile_sample_min_size = 256
            #     vae.vae.tile_latent_min_size = 32
            #     vae.vae.tile_sample_min_size_t = 29
            #     vae.vae.tile_latent_min_size_t = 8
        vae.vae_scale_factor = [4, 8, 8]
        transformer.force_images = False

        # pab
        if config.enable_pab:
            set_pab_manager(config.pab_config)

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
        )

        # cpu offload
        if config.cpu_offload:
            self.enable_model_cpu_offload()
        else:
            self.set_eval_and_device(device, text_encoder, vae, transformer)

        # self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

        # parallel
        self._set_parallel()

    def _set_seed(self, seed):
        if dist.get_world_size() == 1:
            set_seed(seed)
        else:
            set_seed(seed, self.transformer.parallel_manager.dp_rank)

    def _set_parallel(
        self, dp_size: Optional[int] = None, sp_size: Optional[int] = None, enable_cp: Optional[bool] = False
    ):
        # init sequence parallel
        if sp_size is None:
            sp_size = dist.get_world_size()
            dp_size = 1
        else:
            assert (
                dist.get_world_size() % sp_size == 0
            ), f"world_size {dist.get_world_size()} must be divisible by sp_size"
            dp_size = dist.get_world_size() // sp_size

        # transformer parallel
        self.transformer.enable_parallel(dp_size, sp_size, enable_cp)

    # Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/utils.py
    def mask_text_embeddings(self, emb, mask):
        if emb.shape[0] == 1:
            keep_index = mask.sum().item()
            return emb[:, :, :keep_index, :], keep_index  # 1, 120, 4096 -> 1 7 4096
        else:
            masked_feature = emb * mask[:, None, :, None]  # 1 120 4096
            return masked_feature, emb.shape[2]

    # Adapted from diffusers.pipelines.deepfloyd_if.pipeline_if.encode_prompt
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        clean_caption: bool = False,
        mask_feature: bool = True,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For
                PixArt-Alpha, this should be "".
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Alpha, it's should be the embeddings of the ""
                string.
            clean_caption (bool, defaults to `False`):
                If `True`, the function will preprocess and clean the provided caption before encoding.
            mask_feature: (bool, defaults to `True`):
                If `True`, the function will mask the text embeddings.
        """
        embeds_initially_provided = prompt_embeds is not None and negative_prompt_embeds is not None

        if device is None:
            device = self._execution_device

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # See Section 3.1. of the paper.
        max_length = 300

        if prompt_embeds is None:
            prompt = self._text_preprocessing(prompt, clean_caption=clean_caption)
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because the model can only handle sequences up to"
                    f" {max_length} tokens: {removed_text}"
                )

            attention_mask = text_inputs.attention_mask.to(device)
            prompt_embeds_attention_mask = attention_mask

            prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
            prompt_embeds = prompt_embeds[0]
        else:
            prompt_embeds_attention_mask = torch.ones_like(prompt_embeds)

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        prompt_embeds_attention_mask = prompt_embeds_attention_mask.view(bs_embed, -1)
        prompt_embeds_attention_mask = prompt_embeds_attention_mask.repeat(num_images_per_prompt, 1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens = [negative_prompt] * batch_size
            uncond_tokens = self._text_preprocessing(uncond_tokens, clean_caption=clean_caption)
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            attention_mask = uncond_input.attention_mask.to(device)

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
        else:
            negative_prompt_embeds = None

        # print(prompt_embeds.shape) # 1 120 4096
        # print(negative_prompt_embeds.shape) # 1 120 4096

        # Perform additional masking.
        if mask_feature and not embeds_initially_provided:
            prompt_embeds = prompt_embeds.unsqueeze(1)
            masked_prompt_embeds, keep_indices = self.mask_text_embeddings(prompt_embeds, prompt_embeds_attention_mask)
            masked_prompt_embeds = masked_prompt_embeds.squeeze(1)
            masked_negative_prompt_embeds = (
                negative_prompt_embeds[:, :keep_indices, :] if negative_prompt_embeds is not None else None
            )

            # import torch.nn.functional as F

            # padding = (0, 0, 0, 113)  # (左, 右, 下, 上)
            # masked_prompt_embeds_ = F.pad(masked_prompt_embeds, padding, "constant", 0)
            # masked_negative_prompt_embeds_ = F.pad(masked_negative_prompt_embeds, padding, "constant", 0)

            # print(masked_prompt_embeds == masked_prompt_embeds_[:, :masked_negative_prompt_embeds.shape[1], ...])

            return masked_prompt_embeds, masked_negative_prompt_embeds
            # return masked_prompt_embeds_, masked_negative_prompt_embeds_

        return prompt_embeds, negative_prompt_embeds

    def encode_prompt_v120(
        self,
        prompt: Union[str, List[str]],
        do_classifier_free_guidance: bool = True,
        negative_prompt: str = "",
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        clean_caption: bool = False,
        max_sequence_length: int = 120,
        **kwargs,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt not to guide the image generation. If not defined, one has to pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`). For
                PixArt-Alpha, this should be "".
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                whether to use classifier free guidance or not
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                number of images that should be generated per prompt
            device: (`torch.device`, *optional*):
                torch device to place the resulting embeddings on
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Alpha, it's should be the embeddings of the ""
                string.
            clean_caption (`bool`, defaults to `False`):
                If `True`, the function will preprocess and clean the provided caption before encoding.
            max_sequence_length (`int`, defaults to 120): Maximum sequence length to use for the prompt.
        """
        if device is None:
            device = getattr(self, "_execution_device", None) or getattr(self, "device", None) or torch.device("cuda")

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # See Section 3.1. of the paper.
        max_length = max_sequence_length

        if prompt_embeds is None:
            prompt = self._text_preprocessing(prompt, clean_caption=clean_caption)
            text_inputs = self.tokenizer(
                prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {max_length} tokens: {removed_text}"
                )

            prompt_attention_mask = text_inputs.attention_mask
            prompt_attention_mask = prompt_attention_mask.to(device)

            prompt_embeds = self.text_encoder(text_input_ids.to(device), attention_mask=prompt_attention_mask)
            prompt_embeds = prompt_embeds[0]

        if self.text_encoder is not None:
            dtype = self.text_encoder.dtype
        elif self.transformer is not None:
            dtype = self.transformer.dtype
        else:
            dtype = None

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.view(bs_embed, -1)
        prompt_attention_mask = prompt_attention_mask.repeat(num_images_per_prompt, 1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens = [negative_prompt] * batch_size
            uncond_tokens = self._text_preprocessing(uncond_tokens, clean_caption=clean_caption)
            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_attention_mask=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            negative_prompt_attention_mask = uncond_input.attention_mask
            negative_prompt_attention_mask = negative_prompt_attention_mask.to(device)

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device), attention_mask=negative_prompt_attention_mask
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            negative_prompt_attention_mask = negative_prompt_attention_mask.view(bs_embed, -1)
            negative_prompt_attention_mask = negative_prompt_attention_mask.repeat(num_images_per_prompt, 1)
        else:
            negative_prompt_embeds = None
            negative_prompt_attention_mask = None

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_steps,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_attention_mask=None,
        negative_prompt_attention_mask=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )
            if prompt_attention_mask.shape != negative_prompt_attention_mask.shape:
                raise ValueError(
                    "`prompt_attention_mask` and `negative_prompt_attention_mask` must have the same shape when passed directly, but"
                    f" got: `prompt_attention_mask` {prompt_attention_mask.shape} != `negative_prompt_attention_mask`"
                    f" {negative_prompt_attention_mask.shape}."
                )

    # Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._text_preprocessing
    def _text_preprocessing(self, text, clean_caption=False):
        if not isinstance(text, (tuple, list)):
            text = [text]

        def process(text: str):
            if clean_caption:
                text = self._clean_caption(text)
                text = self._clean_caption(text)
            else:
                text = text.lower().strip()
            return text

        return [process(t) for t in text]

    # Copied from diffusers.pipelines.deepfloyd_if.pipeline_if.IFPipeline._clean_caption
    def _clean_caption(self, caption):
        caption = str(caption)
        caption = ul.unquote_plus(caption)
        caption = caption.strip().lower()
        caption = re.sub("<person>", "person", caption)
        # urls:
        caption = re.sub(
            r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
            # noqa
            "",
            caption,
        )  # regex for urls
        caption = re.sub(
            r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
            # noqa
            "",
            caption,
        )  # regex for urls
        # html:
        caption = BeautifulSoup(caption, features="html.parser").text

        # @<nickname>
        caption = re.sub(r"@[\w\d]+\b", "", caption)

        # 31C0—31EF CJK Strokes
        # 31F0—31FF Katakana Phonetic Extensions
        # 3200—32FF Enclosed CJK Letters and Months
        # 3300—33FF CJK Compatibility
        # 3400—4DBF CJK Unified Ideographs Extension A
        # 4DC0—4DFF Yijing Hexagram Symbols
        # 4E00—9FFF CJK Unified Ideographs
        caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
        caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
        caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
        caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
        caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
        caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
        caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)
        #######################################################

        # все виды тире / all types of dash --> "-"
        caption = re.sub(
            r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",
            # noqa
            "-",
            caption,
        )

        # кавычки к одному стандарту
        caption = re.sub(r"[`´«»“”¨]", '"', caption)
        caption = re.sub(r"[‘’]", "'", caption)

        # &quot;
        caption = re.sub(r"&quot;?", "", caption)
        # &amp
        caption = re.sub(r"&amp", "", caption)

        # ip adresses:
        caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)

        # article ids:
        caption = re.sub(r"\d:\d\d\s+$", "", caption)

        # \n
        caption = re.sub(r"\\n", " ", caption)

        # "#123"
        caption = re.sub(r"#\d{1,3}\b", "", caption)
        # "#12345.."
        caption = re.sub(r"#\d{5,}\b", "", caption)
        # "123456.."
        caption = re.sub(r"\b\d{6,}\b", "", caption)
        # filenames:
        caption = re.sub(r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption)

        #
        caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
        caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

        caption = re.sub(self.bad_punct_regex, r" ", caption)  # ***AUSVERKAUFT***, #AUSVERKAUFT
        caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

        # this-is-my-cute-cat / this_is_my_cute_cat
        regex2 = re.compile(r"(?:\-|\_)")
        if len(re.findall(regex2, caption)) > 3:
            caption = re.sub(regex2, " ", caption)

        caption = ftfy.fix_text(caption)
        caption = html.unescape(html.unescape(caption))

        caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
        caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
        caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

        caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
        caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
        caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
        caption = re.sub(r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption)
        caption = re.sub(r"\bpage\s+\d+\b", "", caption)

        caption = re.sub(r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption)  # j2d1a2a...

        caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

        caption = re.sub(r"\b\s+\:\s+", r": ", caption)
        caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
        caption = re.sub(r"\s+", " ", caption)

        caption.strip()

        caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
        caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
        caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
        caption = re.sub(r"^\.\S+$", "", caption)

        return caption.strip()

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        shape = (
            batch_size,
            num_channels_latents,
            (math.ceil((int(num_frames) - 1) / self.vae.vae_scale_factor[0]) + 1)
            if int(num_frames) % 2 == 1
            else math.ceil(int(num_frames) / self.vae.vae_scale_factor[0]),
            math.ceil(int(height) / self.vae.vae_scale_factor[1]),
            math.ceil(int(width) / self.vae.vae_scale_factor[2]),
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    @torch.no_grad()
    def generate(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 100,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        seed: int = -1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        mask_feature: bool = True,
        enable_temporal_attentions: bool = True,
        verbose: bool = True,
        max_sequence_length: int = 512,
    ) -> Union[VideoSysPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_inference_steps (`int`, *optional*, defaults to 100):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process. If not defined, equal spaced `num_inference_steps`
                timesteps are used. Must be in descending order.
            guidance_scale (`float`, *optional*, defaults to 7.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            height (`int`, *optional*, defaults to self.unet.config.sample_size):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size):
                The width in pixels of the generated image.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Alpha this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.IFPipelineOutput`] instead of a plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            clean_caption (`bool`, *optional*, defaults to `True`):
                Whether or not to clean the caption before creating embeddings. Requires `beautifulsoup4` and `ftfy` to
                be installed. If the dependencies are not installed, the embeddings will be created from the raw
                prompt.
            mask_feature (`bool` defaults to `True`): If set to `True`, the text embeddings will be masked.

        Examples:

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        # 1. Check inputs. Raise error if not correct
        height = self.transformer.config.sample_size[0] * self.vae.vae_scale_factor[1]
        width = self.transformer.config.sample_size[1] * self.vae.vae_scale_factor[2]
        num_frames = self._config.num_frames
        update_steps(num_inference_steps)
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_steps,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )
        self._set_seed(seed)

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = getattr(self, "_execution_device", None) or getattr(self, "device", None) or torch.device("cuda")

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        if self._config.version == "v110":
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                clean_caption=clean_caption,
                mask_feature=mask_feature,
            )
            prompt_attention_mask = None
            negative_prompt_attention_mask = None
        else:
            (
                prompt_embeds,
                prompt_attention_mask,
                negative_prompt_embeds,
                negative_prompt_attention_mask,
            ) = self.encode_prompt_v120(
                prompt,
                do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                clean_caption=clean_caption,
                max_sequence_length=max_sequence_length,
            )
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            if prompt_attention_mask is not None and negative_prompt_attention_mask is not None:
                prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)

        # 4. Prepare timesteps
        if self._config.version == "v110":
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
        else:
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, None)

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        # 6.1 Prepare micro-conditions.
        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}

        # 7. Denoising loop
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)

        progress_wrap = tqdm.tqdm if verbose and dist.get_rank() == 0 else (lambda x: x)
        for i, t in progress_wrap(list(enumerate(timesteps))):
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            current_timestep = t
            if not torch.is_tensor(current_timestep):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = latent_model_input.device.type == "mps"
                if isinstance(current_timestep, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latent_model_input.device)
            elif len(current_timestep.shape) == 0:
                current_timestep = current_timestep[None].to(latent_model_input.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            current_timestep = current_timestep.expand(latent_model_input.shape[0])

            if prompt_embeds is not None and prompt_embeds.ndim == 3:
                prompt_embeds = prompt_embeds.unsqueeze(1)  # b l d -> b 1 l d

            if prompt_attention_mask is not None and prompt_attention_mask.ndim == 2:
                prompt_attention_mask = prompt_attention_mask.unsqueeze(1)  # b l -> b 1 l
            # prepare attention_mask.
            # b c t h w -> b t h w
            attention_mask = torch.ones_like(latent_model_input)[:, 0]

            # predict noise model_output
            noise_pred = self.transformer(
                latent_model_input,
                all_timesteps=timesteps,
                encoder_hidden_states=prompt_embeds,
                timestep=current_timestep,
                added_cond_kwargs=added_cond_kwargs,
                enable_temporal_attentions=enable_temporal_attentions,
                attention_mask=attention_mask,
                encoder_attention_mask=prompt_attention_mask,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # learned sigma
            if self.transformer.config.out_channels // 2 == latent_channels:
                noise_pred = noise_pred.chunk(2, dim=1)[0]
            else:
                noise_pred = noise_pred

            # compute previous image: x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                if callback is not None and i % callback_steps == 0:
                    step_idx = i // getattr(self.scheduler, "order", 1)
                    callback(step_idx, t, latents)

        if not output_type == "latents":
            video = self.decode_latents(latents)
            video = video[:, :num_frames, :height, :width]
        else:
            video = latents
            return VideoSysPipelineOutput(video=video)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return VideoSysPipelineOutput(video=video)

    def decode_latents(self, latents):
        video = self.vae(latents.to(self.vae.vae.dtype))
        video = (
            ((video / 2.0 + 0.5).clamp(0, 1) * 255).to(dtype=torch.uint8).cpu().permute(0, 1, 3, 4, 2).contiguous()
        )  # b t h w c
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        return video

    def save_video(self, video, output_path):
        save_video(video, output_path, fps=24)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of the scheduler is used. If `timesteps` is passed, `num_inference_steps`
                must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps
