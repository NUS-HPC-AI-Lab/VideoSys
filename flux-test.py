import os

import torch
from diffusers.models.transformers import FluxTransformer2DModel

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"

# from videosys.pipelines.flux.pipeline_flux_pab import FluxPipeline, FluxConfig
# pipe = FluxPipeline.from_pretrained(
#     "black-forest-labs/FLUX.1-dev",
#     torch_dtype=torch.bfloat16,
#     device_map="balanced",
#     config = FluxConfig()
# )

transformer = FluxTransformer2DModel.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    subfolder="transformer",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    offload_state_dict=False,
).to("cuda:1")

from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16, device_map="balanced"  # 自动分配到多个 GPU
)
# pipe.enable_model_cpu_offload()

prompt = "A cat holding a sign that says hello world"

image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cuda:0").manual_seed(0),
).images[0]

image.save("flux-dev.png")
