import re
from typing import Optional, Tuple, Union

import torch
from diffusers.models import AutoencoderKL

from videosys.core.pab_mgr import PABConfig, set_pab_manager
from videosys.core.pipeline import VideoSysPipeline, VideoSysPipelineOutput
from videosys.utils.utils import save_video

from .datasets import get_image_size, get_num_frames
from .inference_utils import (
    append_generated,
    append_score_to_prompts,
    apply_mask_strategy,
    collect_references_batch,
    dframe_to_frame,
    extract_json_from_prompts,
    extract_prompts_loop,
    merge_prompt,
    prepare_multi_resolution_info,
    split_prompt,
)
from .rflow import RFLOW
from .stdit3 import STDiT3_XL_2
from .text_encoder import T5Encoder, text_preprocessing
from .vae import OpenSoraVAE_V1_2


class OpenSoraPABConfig(PABConfig):
    def __init__(
        self,
        steps: int = 50,
        spatial_broadcast: bool = True,
        spatial_threshold: list = [450, 930],
        spatial_gap: int = 2,
        temporal_broadcast: bool = True,
        temporal_threshold: list = [450, 930],
        temporal_gap: int = 4,
        cross_broadcast: bool = True,
        cross_threshold: list = [450, 930],
        cross_gap: int = 6,
        diffusion_skip: bool = False,
        diffusion_timestep_respacing: list = None,
        diffusion_skip_timestep: list = None,
        mlp_skip: bool = True,
        mlp_spatial_skip_config: dict = {
            676: {"block": [0, 1, 2, 3, 4], "skip_count": 2},
            788: {"block": [0, 1, 2, 3, 4], "skip_count": 2},
            864: {"block": [0, 1, 2, 3, 4], "skip_count": 2},
        },
        mlp_temporal_skip_config: dict = {
            676: {"block": [0, 1, 2, 3, 4], "skip_count": 2},
            788: {"block": [0, 1, 2, 3, 4], "skip_count": 2},
            864: {"block": [0, 1, 2, 3, 4], "skip_count": 2},
        },
    ):
        super().__init__(
            steps=steps,
            spatial_broadcast=spatial_broadcast,
            spatial_threshold=spatial_threshold,
            spatial_gap=spatial_gap,
            temporal_broadcast=temporal_broadcast,
            temporal_threshold=temporal_threshold,
            temporal_gap=temporal_gap,
            cross_broadcast=cross_broadcast,
            cross_threshold=cross_threshold,
            cross_gap=cross_gap,
            diffusion_skip=diffusion_skip,
            diffusion_timestep_respacing=diffusion_timestep_respacing,
            diffusion_skip_timestep=diffusion_skip_timestep,
            mlp_skip=mlp_skip,
            mlp_spatial_skip_config=mlp_spatial_skip_config,
            mlp_temporal_skip_config=mlp_temporal_skip_config,
        )


class OpenSoraConfig:
    def __init__(
        self,
        world_size: int = 1,
        transformer: str = "hpcai-tech/OpenSora-STDiT-v3",
        vae: str = "hpcai-tech/OpenSora-VAE-v1.2",
        text_encoder: str = "DeepFloyd/t5-v1_1-xxl",
        # ======= scheduler =======
        num_sampling_steps: int = 30,
        cfg_scale: float = 7.0,
        # ======= vae ========
        tiling_size: int = 4,
        # ======= pab ========
        enable_pab: bool = False,
        pab_config: PABConfig = OpenSoraPABConfig(),
    ):
        # ======= engine ========
        self.world_size = world_size

        # ======= pipeline ========
        self.pipeline_cls = OpenSoraPipeline
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder

        # ======= scheduler ========
        self.num_sampling_steps = num_sampling_steps
        self.cfg_scale = cfg_scale

        # ======= vae ========
        self.tiling_size = tiling_size

        # ======= pab ========
        self.enable_pab = enable_pab
        self.pab_config = pab_config


class OpenSoraPipeline(VideoSysPipeline):
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

    _optional_components = ["tokenizer", "text_encoder"]
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    def __init__(
        self,
        config: OpenSoraConfig,
        text_encoder: Optional[T5Encoder] = None,
        vae: Optional[AutoencoderKL] = None,
        transformer: Optional[STDiT3_XL_2] = None,
        scheduler: Optional[RFLOW] = None,
        device: torch.device = torch.device("cuda"),
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self._config = config
        self._device = device
        self._dtype = dtype

        # initialize the model if not provided
        if text_encoder is None:
            text_encoder = T5Encoder(
                from_pretrained=config.text_encoder, model_max_length=300, device=device, dtype=dtype
            )
        if vae is None:
            vae = OpenSoraVAE_V1_2(
                from_pretrained="hpcai-tech/OpenSora-VAE-v1.2",
                micro_frame_size=17,
                micro_batch_size=config.tiling_size,
            ).to(dtype)
        if transformer is None:
            transformer = STDiT3_XL_2(
                from_pretrained="hpcai-tech/OpenSora-STDiT-v3",
                qk_norm=True,
                enable_flash_attn=True,
                enable_layernorm_kernel=True,
                in_channels=vae.out_channels,
                caption_channels=text_encoder.output_dim,
                model_max_length=text_encoder.model_max_length,
            ).to(device, dtype)
            text_encoder.y_embedder = transformer.y_embedder
        if scheduler is None:
            scheduler = RFLOW(
                use_timestep_transform=True, num_sampling_steps=config.num_sampling_steps, cfg_scale=config.cfg_scale
            )

        # pab
        if config.enable_pab:
            set_pab_manager(config.pab_config)

        # set eval and device
        self.set_eval_and_device(device, text_encoder, vae, transformer)

        self.register_modules(text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        resolution="480p",
        aspect_ratio="9:16",
        num_frames: int = 51,
        loop: int = 1,
        llm_refine: bool = False,
        negative_prompt: str = "",
        ms: Optional[str] = "",
        refs: Optional[str] = "",
        aes: float = 6.5,
        flow: Optional[float] = None,
        camera_motion: Optional[float] = None,
        condition_frame_length: int = 5,
        align: int = 5,
        condition_frame_edit: float = 0.0,
        return_dict: bool = True,
        verbose: bool = True,
    ) -> Union[VideoSysPipelineOutput, Tuple]:
        """
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            resolution (`str`, *optional*, defaults to `"480p"`):
                The resolution of the generated video.
            aspect_ratio (`str`, *optional*, defaults to `"9:16"`):
                The aspect ratio of the generated video.
            num_frames (`int`, *optional*, defaults to 51):
                The number of frames to generate.
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
        # == basic ==
        fps = 24
        image_size = get_image_size(resolution, aspect_ratio)
        num_frames = get_num_frames(num_frames)

        # == prepare batch prompts ==
        batch_prompts = [prompt]
        ms = [ms]
        refs = [refs]

        # == get json from prompts ==
        batch_prompts, refs, ms = extract_json_from_prompts(batch_prompts, refs, ms)

        # == get reference for condition ==
        refs = collect_references_batch(refs, self.vae, image_size)

        # == multi-resolution info ==
        model_args = prepare_multi_resolution_info(
            "OpenSora", len(batch_prompts), image_size, num_frames, fps, self._device, self._dtype
        )

        # == process prompts step by step ==
        # 0. split prompt
        # each element in the list is [prompt_segment_list, loop_idx_list]
        batched_prompt_segment_list = []
        batched_loop_idx_list = []
        for prompt in batch_prompts:
            prompt_segment_list, loop_idx_list = split_prompt(prompt)
            batched_prompt_segment_list.append(prompt_segment_list)
            batched_loop_idx_list.append(loop_idx_list)

        # 1. refine prompt by openai
        # if llm_refine:
        # only call openai API when
        # 1. seq parallel is not enabled
        # 2. seq parallel is enabled and the process is rank 0
        # if not enable_sequence_parallelism or (enable_sequence_parallelism and coordinator.is_master()):
        #     for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
        #         batched_prompt_segment_list[idx] = refine_prompts_by_openai(prompt_segment_list)

        # # sync the prompt if using seq parallel
        # if enable_sequence_parallelism:
        #     coordinator.block_all()
        #     prompt_segment_length = [
        #         len(prompt_segment_list) for prompt_segment_list in batched_prompt_segment_list
        #     ]

        #     # flatten the prompt segment list
        #     batched_prompt_segment_list = [
        #         prompt_segment
        #         for prompt_segment_list in batched_prompt_segment_list
        #         for prompt_segment in prompt_segment_list
        #     ]

        #     # create a list of size equal to world size
        #     broadcast_obj_list = [batched_prompt_segment_list] * coordinator.world_size
        #     dist.broadcast_object_list(broadcast_obj_list, 0)

        #     # recover the prompt list
        #     batched_prompt_segment_list = []
        #     segment_start_idx = 0
        #     all_prompts = broadcast_obj_list[0]
        #     for num_segment in prompt_segment_length:
        #         batched_prompt_segment_list.append(
        #             all_prompts[segment_start_idx : segment_start_idx + num_segment]
        #         )
        #         segment_start_idx += num_segment

        # 2. append score
        for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
            batched_prompt_segment_list[idx] = append_score_to_prompts(
                prompt_segment_list,
                aes=aes,
                flow=flow,
                camera_motion=camera_motion,
            )

        # 3. clean prompt with T5
        for idx, prompt_segment_list in enumerate(batched_prompt_segment_list):
            batched_prompt_segment_list[idx] = [text_preprocessing(prompt) for prompt in prompt_segment_list]

        # 4. merge to obtain the final prompt
        batch_prompts = []
        for prompt_segment_list, loop_idx_list in zip(batched_prompt_segment_list, batched_loop_idx_list):
            batch_prompts.append(merge_prompt(prompt_segment_list, loop_idx_list))

        # == Iter over loop generation ==
        video_clips = []
        for loop_i in range(loop):
            # == get prompt for loop i ==
            batch_prompts_loop = extract_prompts_loop(batch_prompts, loop_i)

            # == add condition frames for loop ==
            if loop_i > 0:
                refs, ms = append_generated(
                    self.vae, video_clips[-1], refs, ms, loop_i, condition_frame_length, condition_frame_edit
                )

            # == sampling ==
            input_size = (num_frames, *image_size)
            latent_size = self.vae.get_latent_size(input_size)
            z = torch.randn(
                len(batch_prompts), self.vae.out_channels, *latent_size, device=self._device, dtype=self._dtype
            )
            masks = apply_mask_strategy(z, refs, ms, loop_i, align=align)
            samples = self.scheduler.sample(
                self.transformer,
                self.text_encoder,
                z=z,
                prompts=batch_prompts_loop,
                device=self._device,
                additional_args=model_args,
                progress=verbose,
                mask=masks,
            )
            samples = self.vae.decode(samples.to(self._dtype), num_frames=num_frames)
            video_clips.append(samples)

        for i in range(1, loop):
            video_clips[i] = video_clips[i][:, dframe_to_frame(condition_frame_length) :]
        video = torch.cat(video_clips, dim=1)

        low, high = -1, 1
        video.clamp_(min=low, max=high)
        video.sub_(low).div_(max(high - low, 1e-5))
        video = video.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 4, 1).to("cpu", torch.uint8)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return VideoSysPipelineOutput(video=video)

    def save_video(self, video, output_path):
        save_video(video, output_path, fps=24)
