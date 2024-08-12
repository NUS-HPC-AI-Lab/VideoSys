import os

import deltadit
from deltadit import OpenSoraConfig, OpenSoraDELTAConfig, OpenSoraPipeline


def run_base():
    # Manually set environment variables for single GPU debugging
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"

    deltadit.initialize(42)

    config = OpenSoraConfig()
    pipeline = OpenSoraPipeline(config)

    prompt = "Yellow and black tropical fish dart through the sea."
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/{prompt}.mp4")


def run_pab():
    # Manually set environment variables for single GPU debugging
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12356"

    deltadit.initialize(42)

    delta_config = OpenSoraDELTAConfig(
        steps=10,
        delta_skip=True,
        # delta_threshold={(0, 10): [0, 9]},
        delta_threshold={(0, 10): [0, 1], (20, 30): [9, 10]},
        # delta_threshold={(0,10):[0,9], (20,30):[9,27]},
        # delta_threshold={(0,10):[9,27], (20,30):[9,27]},
        delta_gap=2,
    )

    config = OpenSoraConfig(enable_delta=True, delta_config=delta_config)
    pipeline = OpenSoraPipeline(config)

    prompt = "Yellow and black tropical fish dart through the sea."
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/{prompt}.mp4")
    print(f"Done ./outputs/{prompt}.mp4")


if __name__ == "__main__":
    # run_base()
    run_pab()
