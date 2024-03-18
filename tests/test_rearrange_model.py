import json
import os

import colossalai
import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.booster.plugin import TorchDDPPlugin
from colossalai.cluster import DistCoordinator
from colossalai.testing import rerun_if_address_is_in_use, spawn
from colossalai.utils import get_current_device
from torch.testing import assert_close

from opendit.models.dit.dit import DiT, DiT_models
from opendit.utils.ckpt_utils import load, record_model_param_shape, save
from opendit.utils.train_utils import requires_grad

WORKERS = 2


def sequence_parallel_ckpt_load():
    DTYPE = torch.float32
    device = get_current_device()
    torch.manual_seed(1024)

    # initialize booster
    booster_origin = Booster(plugin=TorchDDPPlugin())
    booster_sp = Booster(plugin=TorchDDPPlugin())

    coordinator = DistCoordinator()

    optimizer = None
    lr_scheduler = None
    dataloader = None

    model_type = "DiT-S/2"
    image_size = 256
    latent_size = image_size // 8
    num_classes = 1000
    enable_layernorm_kernel = True
    enable_modulate_kernel = True
    enable_flashattn = False
    sequence_parallel_size = WORKERS
    sequence_parallel_type = "longseq"
    torch.set_default_dtype(DTYPE)

    model_config = {
        "input_size": latent_size,
        "num_classes": num_classes,
        "enable_layernorm_kernel": enable_layernorm_kernel,
        "enable_modulate_kernel": enable_modulate_kernel,
        "enable_flashattn": enable_flashattn,
    }

    # Create an experiment folder
    output_dir = "./outputs"
    experiment_dir_origin = f"{output_dir}/load-origin-model"
    dist.barrier()
    if coordinator.is_master():
        os.makedirs(experiment_dir_origin, exist_ok=True)
        with open(f"{experiment_dir_origin}/config.txt", "w") as f:
            json.dump(model_config, f, indent=4)
        print(f"Experiment directory created at {experiment_dir_origin}")

    experiment_dir_sp = f"{output_dir}/load-sp-model"
    dist.barrier()
    if coordinator.is_master():
        os.makedirs(experiment_dir_sp, exist_ok=True)
        with open(f"{experiment_dir_sp}/config.txt", "w") as f:
            json.dump(model_config, f, indent=4)
        print(f"Experiment directory created at {experiment_dir_sp}")

    # A DiT model whose sp type is None
    model_origin: DiT = DiT_models[model_type](**model_config).to(device)

    ema_origin: DiT = DiT_models[model_type](**model_config).to(device)
    ema_origin = ema_origin.to(DTYPE)
    ema_origin.load_state_dict(model_origin.state_dict())
    requires_grad(ema_origin, False)
    ema_shape_dict_origin = record_model_param_shape(ema_origin)

    model_origin, _, _, _, _ = booster_origin.boost(
        model=model_origin, optimizer=optimizer, lr_scheduler=lr_scheduler, dataloader=dataloader
    )

    # A DiT model whose sp type is "longseq"
    model_sp: DiT = DiT_models[model_type](
        sequence_parallel_size=sequence_parallel_size,
        sequence_parallel_group=None,
        sequence_parallel_type=sequence_parallel_type,
        **model_config,
    ).to(device)

    ema_sp: DiT = DiT_models[model_type](**model_config).to(device)
    ema_sp = ema_sp.to(DTYPE)
    ema_sp.load_state_dict(model_sp.state_dict())
    requires_grad(ema_sp, False)
    record_model_param_shape(ema_sp)

    model_sp, _, _, _, _ = booster_sp.boost(
        model=model_sp, optimizer=optimizer, lr_scheduler=lr_scheduler, dataloader=dataloader
    )

    # Prepare data
    x = torch.randn(2, 4, 32, 32).cuda()
    t = torch.randn(2).to(torch.int64).cuda()
    y = torch.randn(2).to(torch.int64).cuda()

    # (1) save origin model
    if dist.get_rank() == 0:
        print("Origin model begin to save")

    save(
        booster_origin,
        model_origin,
        ema_origin,
        optimizer,
        lr_scheduler,
        0,
        0 + 1,
        0 + 1,
        2,
        coordinator,
        experiment_dir_origin,
        ema_shape_dict_origin,
        sequence_parallel_type=None,
    )
    if dist.get_rank() == 0:
        print("Origin model saved")

    # (2) load origin model's weight to the sp model
    if dist.get_rank() == 0:
        print("Origin model begin to load")
    start_epoch, start_step, sampler_start_idx = load(
        booster_sp,
        model_sp,
        ema_sp,
        optimizer,
        lr_scheduler,
        f"{experiment_dir_origin}/epoch{0}-global_step{1}",
        sequence_parallel_type="longseq",
    )
    if dist.get_rank() == 0:
        print("Origin model load to sp model")

    # (3) check whether the two model have the same forward results
    origin_output = model_origin(x, t, y)
    sp_output = model_sp(x, t, y)
    assert_close(origin_output, sp_output, atol=5e-5, rtol=5e-5)


def sequence_parallel_ckpt_save():
    DTYPE = torch.float32
    device = get_current_device()
    torch.manual_seed(1024)

    # initialize booster
    booster_origin = Booster(plugin=TorchDDPPlugin())
    booster_sp = Booster(plugin=TorchDDPPlugin())

    coordinator = DistCoordinator()

    optimizer = None
    lr_scheduler = None
    dataloader = None

    model_type = "DiT-S/2"
    image_size = 256
    latent_size = image_size // 8
    num_classes = 1000
    enable_layernorm_kernel = True
    enable_modulate_kernel = True
    enable_flashattn = False
    sequence_parallel_size = WORKERS
    sequence_parallel_type = "longseq"
    torch.set_default_dtype(DTYPE)

    model_config = {
        "input_size": latent_size,
        "num_classes": num_classes,
        "enable_layernorm_kernel": enable_layernorm_kernel,
        "enable_modulate_kernel": enable_modulate_kernel,
        "enable_flashattn": enable_flashattn,
    }

    # Create an experiment folder
    output_dir = "./outputs"
    experiment_dir_origin = f"{output_dir}/save-origin-model"
    dist.barrier()
    if coordinator.is_master():
        os.makedirs(experiment_dir_origin, exist_ok=True)
        with open(f"{experiment_dir_origin}/config.txt", "w") as f:
            json.dump(model_config, f, indent=4)
        print(f"Experiment directory created at {experiment_dir_origin}")

    experiment_dir_sp = f"{output_dir}/save-sp-model"
    dist.barrier()
    if coordinator.is_master():
        os.makedirs(experiment_dir_sp, exist_ok=True)
        with open(f"{experiment_dir_sp}/config.txt", "w") as f:
            json.dump(model_config, f, indent=4)
        print(f"Experiment directory created at {experiment_dir_sp}")

    # A DiT model whose sp type is None
    model_origin: DiT = DiT_models[model_type](**model_config).to(device)

    ema_origin: DiT = DiT_models[model_type](**model_config).to(device)
    ema_origin = ema_origin.to(DTYPE)
    ema_origin.load_state_dict(model_origin.state_dict())
    requires_grad(ema_origin, False)
    record_model_param_shape(ema_origin)

    model_origin, _, _, _, _ = booster_origin.boost(
        model=model_origin, optimizer=optimizer, lr_scheduler=lr_scheduler, dataloader=dataloader
    )

    # A DiT model whose sp type is "longseq"
    model_sp: DiT = DiT_models[model_type](
        sequence_parallel_size=sequence_parallel_size,
        sequence_parallel_group=None,
        sequence_parallel_type=sequence_parallel_type,
        **model_config,
    ).to(device)

    ema_sp: DiT = DiT_models[model_type](**model_config).to(device)
    ema_sp = ema_sp.to(DTYPE)
    ema_sp.load_state_dict(model_sp.state_dict())
    requires_grad(ema_sp, False)
    ema_shape_dict_sp = record_model_param_shape(ema_sp)

    model_sp, _, _, _, _ = booster_sp.boost(
        model=model_sp, optimizer=optimizer, lr_scheduler=lr_scheduler, dataloader=dataloader
    )

    # Prepare data
    x = torch.randn(2, 4, 32, 32).cuda()
    t = torch.randn(2).to(torch.int64).cuda()
    y = torch.randn(2).to(torch.int64).cuda()

    # (4) save sp model
    if dist.get_rank() == 0:
        print("Sp model begin to save")

    save(
        booster_sp,
        model_sp,
        ema_sp,
        optimizer,
        lr_scheduler,
        0,
        0 + 1,
        0 + 1,
        2,
        coordinator,
        experiment_dir_sp,
        ema_shape_dict_sp,
        sequence_parallel_type="longseq",
    )

    if dist.get_rank() == 0:
        print("Sp model saved")

    # (5) load sp model's weight to the origin model
    if dist.get_rank() == 0:
        print("Origin model begin to load")

    start_epoch, start_step, sampler_start_idx = load(
        booster_origin,
        model_origin,
        ema_origin,
        optimizer,
        lr_scheduler,
        f"{experiment_dir_sp}/epoch{0}-global_step{1}",
        sequence_parallel_type=None,
    )

    if dist.get_rank() == 0:
        print("Sp model load to origin model")

    # (6) check whether the two model have the same forward results
    origin_output = model_origin(x, t, y)
    sp_output = model_sp(x, t, y)
    assert_close(origin_output, sp_output, atol=5e-5, rtol=5e-5)


def run_sp_ckpt_load():
    sequence_parallel_ckpt_load()


def check_longseq_ckpt_load(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_sp_ckpt_load()


@rerun_if_address_is_in_use()
def test_sequence_parallel_ckpt_load():
    spawn(check_longseq_ckpt_load, nprocs=WORKERS)


def run_sp_ckpt_save():
    sequence_parallel_ckpt_save()


def check_longseq_ckpt_save(rank, world_size, port):
    colossalai.launch(config={}, rank=rank, world_size=world_size, host="localhost", port=port, backend="nccl")
    run_sp_ckpt_save()


@rerun_if_address_is_in_use()
def test_sequence_parallel_ckpt_save():
    spawn(check_longseq_ckpt_save, nprocs=WORKERS)


if __name__ == "__main__":
    test_sequence_parallel_ckpt_load()
    test_sequence_parallel_ckpt_save()
