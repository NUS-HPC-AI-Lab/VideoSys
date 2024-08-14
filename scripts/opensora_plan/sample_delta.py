import deltadit
from deltadit import OpenSoraPlanConfig, OpenSoraPlanDELTAConfig, OpenSoraPlanPipeline


def run_base():
    # Manually set environment variables for single GPU debugging
    # os.environ["RANK"] = "0"
    # os.environ["LOCAL_RANK"] = "0"
    # os.environ["WORLD_SIZE"] = "1"
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12359"

    deltadit.initialize(42)

    config = OpenSoraPlanConfig()
    pipeline = OpenSoraPlanPipeline(config)
    for i in [1, 2]:
        print(f"Running iteration {i}")
        prompt = "a bear hunting for prey"
        video = pipeline.generate(prompt).video[0]
        pipeline.save_video(video, f"./outputs/opensora_plan_delta_base_{prompt}.mp4")


def run_pab():
    # Manually set environment variables for single GPU debugging
    # os.environ["RANK"] = "0"
    # os.environ["LOCAL_RANK"] = "0"
    # os.environ["WORLD_SIZE"] = "1"
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"

    deltadit.initialize(42)

    delta_config = OpenSoraPlanDELTAConfig(
        steps=10,
        delta_skip=True,
        delta_threshold={(0, 1): [0, 2]},
        # delta_threshold={(0, 6): [0, 2], (24, 30): [25, 27]},
        delta_gap=2,
    )
    config = OpenSoraPlanConfig(enable_delta=True, delta_config=delta_config)
    pipeline = OpenSoraPlanPipeline(config)

    prompt = "a bear hunting for prey"
    video = pipeline.generate(prompt).video[0]

    save_path = f"./outputs/opensora_plan_delta_acc_{config.delta_config.delta_skip}_{prompt.replace(' ', '_')}_delta_threshold_{config.delta_config.delta_threshold}_delta_gap_{config.delta_config.delta_gap}.mp4"
    pipeline.save_video(video, save_path)
    print(f"Saved video to {save_path}")


if __name__ == "__main__":
    # run_base() # 02:59
    run_pab()  # 02:58

# CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 scripts/opensora_plan/sample_delta.py
