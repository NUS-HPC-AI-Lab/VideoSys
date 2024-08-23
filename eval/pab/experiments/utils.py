import json
import os

import tqdm

from videosys.utils.utils import set_seed


def generate_func(pipeline, prompt_list, output_dir, loop: int = 5, kwargs: dict = {}):
    kwargs["verbose"] = False
    for prompt in tqdm.tqdm(prompt_list):
        for l in range(loop):
            set_seed(l)
            video = pipeline.generate(prompt, **kwargs).video[0]
            pipeline.save_video(video, os.path.join(output_dir, f"{prompt}-{l}.mp4"))


def read_prompt_list(prompt_list_path):
    with open(prompt_list_path, "r") as f:
        prompt_list = json.load(f)
    prompt_list = [prompt["prompt_en"] for prompt in prompt_list]
    return prompt_list
