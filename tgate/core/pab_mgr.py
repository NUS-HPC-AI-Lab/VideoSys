import random

import numpy as np
import torch

from tgate.utils.logging import logger

PAB_MANAGER = None


class TGATEConfig:
    def __init__(
        self,
        steps: int,
        cross_broadcast: bool,
        cross_threshold: list,
        cross_gap: int,
        spatial_broadcast: bool,
        spatial_threshold: list,
        spatial_gap: int,
        temporal_broadcast: bool,
        temporal_threshold: list,
        temporal_gap: int,
        diffusion_skip: bool,
        diffusion_timestep_respacing: list,
        diffusion_skip_timestep: list,
    ):
        self.steps = steps

        self.cross_broadcast = cross_broadcast
        self.cross_threshold = cross_threshold
        self.cross_gap = cross_gap

        self.spatial_broadcast = spatial_broadcast
        self.spatial_threshold = spatial_threshold
        self.spatial_gap = spatial_gap

        self.temporal_broadcast = temporal_broadcast
        self.temporal_threshold = temporal_threshold
        self.temporal_gap = temporal_gap

        self.diffusion_skip = diffusion_skip
        self.diffusion_timestep_respacing = diffusion_timestep_respacing
        self.diffusion_skip_timestep = diffusion_skip_timestep


class TGATEManager:
    def __init__(self, config: TGATEConfig):
        self.config: TGATEConfig = config

        init_prompt = f"Init TGATEManager. steps: {config.steps}."
        init_prompt += f" spatial_broadcast: {config.spatial_broadcast}, spatial_threshold: {config.spatial_threshold}, spatial_gap: {config.spatial_gap}."
        init_prompt += f" temporal_broadcast: {config.temporal_broadcast}, temporal_threshold: {config.temporal_threshold}, temporal_gap: {config.temporal_gap}."
        init_prompt += f" cross_broadcast: {config.cross_broadcast}, cross_threshold: {config.cross_threshold}, cross_gap: {config.cross_gap}."
        logger.info(init_prompt)

    def if_broadcast_cross(self, timestep: int, count: int):
        if (
            self.config.cross_broadcast
            and (timestep is not None)
            and (count % self.config.cross_gap != 0)
            and (self.config.cross_threshold[0] < timestep < self.config.cross_threshold[1])
        ):
            flag = True
        else:
            flag = False
        count = (count + 1) % self.config.steps
        return flag, count

    def if_broadcast_temporal(self, timestep: int, count: int):
        if (
            self.config.temporal_broadcast
            and (timestep is not None)
            and (count % self.config.temporal_gap != 0)
            and (self.config.temporal_threshold[0] < timestep < self.config.temporal_threshold[1])
        ):
            flag = True
        else:
            flag = False
        count = (count + 1) % self.config.steps
        return flag, count

    def if_broadcast_spatial(self, timestep: int, count: int, block_idx: int):
        if (
            self.config.spatial_broadcast
            and (timestep is not None)
            and (count % self.config.spatial_gap != 0)
            and (self.config.spatial_threshold[0] < timestep < self.config.spatial_threshold[1])
        ):
            flag = True
        else:
            flag = False
        count = (count + 1) % self.config.steps
        return flag, count


def set_tgate_manager(config: TGATEConfig):
    global PAB_MANAGER
    PAB_MANAGER = TGATEManager(config)


def enable_tgate():
    if PAB_MANAGER is None:
        return False
    return (
        PAB_MANAGER.config.cross_broadcast
        or PAB_MANAGER.config.spatial_broadcast
        or PAB_MANAGER.config.temporal_broadcast
    )


def update_steps(steps: int):
    if PAB_MANAGER is not None:
        PAB_MANAGER.config.steps = steps


def if_broadcast_cross(timestep: int, count: int):
    if not enable_tgate():
        return False, count
    return PAB_MANAGER.if_broadcast_cross(timestep, count)


def if_broadcast_temporal(timestep: int, count: int):
    if not enable_tgate():
        return False, count
    return PAB_MANAGER.if_broadcast_temporal(timestep, count)


def if_broadcast_spatial(timestep: int, count: int, block_idx: int):
    if not enable_tgate():
        return False, count
    return PAB_MANAGER.if_broadcast_spatial(timestep, count, block_idx)


def get_diffusion_skip():
    return enable_tgate() and PAB_MANAGER.config.diffusion_skip


def get_diffusion_timestep_respacing():
    return PAB_MANAGER.config.diffusion_timestep_respacing


def get_diffusion_skip_timestep():
    return enable_tgate() and PAB_MANAGER.config.diffusion_skip_timestep


def space_timesteps(time_steps, time_bins):
    num_bins = len(time_bins)
    bin_size = time_steps // num_bins

    result = []

    for i, bin_count in enumerate(time_bins):
        start = i * bin_size
        end = start + bin_size

        bin_steps = np.linspace(start, end, bin_count, endpoint=False, dtype=int).tolist()
        result.extend(bin_steps)

    result_tensor = torch.tensor(result, dtype=torch.int32)
    sorted_tensor = torch.sort(result_tensor, descending=True).values

    return sorted_tensor


def skip_diffusion_timestep(timesteps, diffusion_skip_timestep):
    if isinstance(timesteps, list):
        # If timesteps is a list, we assume each element is a tensor
        timesteps_np = [t.cpu().numpy() for t in timesteps]
        device = timesteps[0].device
    else:
        # If timesteps is a tensor
        timesteps_np = timesteps.cpu().numpy()
        device = timesteps.device

    num_bins = len(diffusion_skip_timestep)

    if isinstance(timesteps_np, list):
        bin_size = len(timesteps_np) // num_bins
        new_timesteps = []

        for i in range(num_bins):
            bin_start = i * bin_size
            bin_end = (i + 1) * bin_size if i != num_bins - 1 else len(timesteps_np)
            bin_timesteps = timesteps_np[bin_start:bin_end]

            if diffusion_skip_timestep[i] == 0:
                # If the bin is marked with 0, keep all timesteps
                new_timesteps.extend(bin_timesteps)
            elif diffusion_skip_timestep[i] == 1:
                # If the bin is marked with 1, omit the last timestep in the bin
                new_timesteps.extend(bin_timesteps[1:])

        new_timesteps_tensor = [torch.tensor(t, device=device) for t in new_timesteps]
    else:
        bin_size = len(timesteps_np) // num_bins
        new_timesteps = []

        for i in range(num_bins):
            bin_start = i * bin_size
            bin_end = (i + 1) * bin_size if i != num_bins - 1 else len(timesteps_np)
            bin_timesteps = timesteps_np[bin_start:bin_end]

            if diffusion_skip_timestep[i] == 0:
                # If the bin is marked with 0, keep all timesteps
                new_timesteps.extend(bin_timesteps)
            elif diffusion_skip_timestep[i] == 1:
                # If the bin is marked with 1, omit the last timestep in the bin
                new_timesteps.extend(bin_timesteps[1:])
            elif diffusion_skip_timestep[i] != 0:
                # If the bin is marked with a non-zero value, randomly omit n timesteps
                if len(bin_timesteps) > diffusion_skip_timestep[i]:
                    indices_to_remove = set(random.sample(range(len(bin_timesteps)), diffusion_skip_timestep[i]))
                    timesteps_to_keep = [
                        timestep for idx, timestep in enumerate(bin_timesteps) if idx not in indices_to_remove
                    ]
                else:
                    timesteps_to_keep = bin_timesteps  # 如果bin_timesteps的长度小于等于n，则不删除任何元素
                new_timesteps.extend(timesteps_to_keep)

        new_timesteps_tensor = torch.tensor(new_timesteps, device=device)

    if isinstance(timesteps, list):
        return new_timesteps_tensor
    else:
        return new_timesteps_tensor
