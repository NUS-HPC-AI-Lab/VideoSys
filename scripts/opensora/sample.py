import multiprocessing

import opendit
from opendit import OpenSoraConfig, OpenSoraPipeline
from opendit.utils.utils import get_distributed_init_method, get_open_port

mp = multiprocessing.get_context("spawn")


def run_base(rank=0, world_size=1, init_method=None):
    opendit.initialize(rank, world_size, init_method, 42)

    config = OpenSoraConfig()
    pipeline = OpenSoraPipeline(config)

    prompt = "Sunset over the sea."
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/{prompt}.mp4")


def run_pab(rank=0, world_size=1, init_method=None):
    opendit.initialize(rank, world_size, init_method, 42)

    config = OpenSoraConfig(enable_pab=True)
    pipeline = OpenSoraPipeline(config)

    prompt = "Sunset over the sea."
    video = pipeline.generate(prompt).video[0]
    pipeline.save_video(video, f"./outputs/{prompt}.mp4")


def run(world_size=2):
    if world_size == 1:
        run_base()
    else:
        init_method = get_distributed_init_method(
            "127.0.0.1", get_open_port())
        workers = []
        for rank in range(1, world_size):
            worker = mp.Process(
                    target=run_base,
                    kwargs=dict(
                        rank=rank,
                        world_size=world_size,
                        init_method=init_method,
                    )
            )
            worker.start()
            workers.append(worker)

        run_base(world_size=world_size, init_method=init_method)

        for worker in workers:
            worker.join()


if __name__ == "__main__":
    # run_base()
    # run_pab()
    run()
