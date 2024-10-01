import csv
import os

import tqdm


def load_eval_prompts(csv_file_path):
    prompts_dict = {}
    # Read the CSV file
    with open(csv_file_path, mode="r", newline="", encoding="utf-8") as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            prompts_dict[row["id"]] = row["text"]
    return prompts_dict


def generate_func(pipeline, prompt_list, output_dir, kwargs: dict = {}):
    kwargs["verbose"] = False
    kwargs["seed"] = 0
    for idx, prompt in tqdm.tqdm(list(prompt_list.items())):
        if os.path.exists(os.path.join(output_dir, f"{idx}.mp4")):
            print(f"Skip {idx} because it already exists")
            continue
        video = pipeline.generate(prompt, **kwargs).video[0]
        pipeline.save_video(video, os.path.join(output_dir, f"{idx}.mp4"))
