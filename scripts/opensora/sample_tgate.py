import tgate
from tgate import OpenSoraConfig, OpenSoraPipeline, OpenSoraTGATEConfig


def run_base():
    # Manually set environment variables for single GPU debugging
    # os.environ["RANK"] = "0"
    # os.environ["LOCAL_RANK"] = "0"
    # os.environ["WORLD_SIZE"] = "1"
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"

    tgate.initialize(42)

    config = OpenSoraConfig()
    pipeline = OpenSoraPipeline(config)

    prompt = "a bear hunting for prey"
    prompt = "a bear hunting for prey"
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/opensora_tgate_base_{prompt}.mp4")


def run_pab():
    # Manually set environment variables for single GPU debugging
    # os.environ["RANK"] = "0"
    # os.environ["LOCAL_RANK"] = "0"
    # os.environ["WORLD_SIZE"] = "1"
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12358"

    tgate.initialize(42)

    tgate_config = OpenSoraTGATEConfig(
        spatial_broadcast=True,
        spatial_threshold=[0, 12],
        spatial_gap=2,
        temporal_broadcast=True,
        temporal_threshold=[0, 12],
        temporal_gap=2,
        cross_broadcast=True,
        cross_threshold=[12, 30],
        cross_gap=18,
        cross_gap=18,
    )
    # step 250 / m=100 / k=10
    # opensora step=30 / m=12 / k=2
    # latte step=50 / m=20 / k=2
    prompt = "a bear hunting for prey"
    config = OpenSoraConfig(enable_tgate=True, tgate_config=tgate_config)
    save_path = f"./outputs/opensora_tgate_acc_{prompt.replace(' ', '_')}_spatial_{config.tgate_config.spatial_broadcast}_{config.tgate_config.spatial_threshold}_cross_{config.tgate_config.cross_broadcast}_{config.tgate_config.cross_threshold}.mp4"
    pipeline = OpenSoraPipeline(config)

    video = pipeline.generate(prompt).video[0]

    pipeline.save_video(video, save_path)
    print(f"Saved video to {save_path}")


if __name__ == "__main__":
    # torch.backends.cudnn.enabled = False

    # run_base() # 01:18
    run_pab()  # 01:07

# CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 scripts/opensora/sample_tgate.py
