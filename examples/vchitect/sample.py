import os
import random
from io import BytesIO
from typing import List

import imageio
import numpy as np
import torch
from PIL import Image

from videosys import VchitectXLPipeline


def images_to_mp4_bytes(images: List[Image.Image], duration: float = 1000) -> bytes:
    with BytesIO() as output_buffer:
        with imageio.get_writer(output_buffer, format="mp4", fps=1 / (duration / 1000)) as writer:
            for img in images:
                writer.append_data(np.array(img))
        mp4_bytes = output_buffer.getvalue()
    return mp4_bytes


def save_as_mp4(images: List[Image.Image], file_path: str, duration: float = 1000):
    with open(file_path, "wb") as f:
        f.write(images_to_mp4_bytes(images, duration))


def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def infer():
    pipe = VchitectXLPipeline("Vchitect/Vchitect-2.0-2B")
    idx = 0

    set_seed(0)
    prompt = ["sunset over the sea."]
    video = pipe(
        prompt,
        negative_prompt="",
        num_inference_steps=100,
        guidance_scale=7.5,
        width=768,
        height=432,  # 480x288  624x352 432x240 768x432
        frames=40,
    )

    images = video

    duration = 1000 / 8

    save_dir = "./results"
    os.makedirs(save_dir, exist_ok=True)

    idx += 1

    save_as_mp4(images, os.path.join(save_dir, f"sample_{idx}_seed{1}") + ".mp4", duration=duration)


def main():
    infer()


if __name__ == "__main__":
    main()
