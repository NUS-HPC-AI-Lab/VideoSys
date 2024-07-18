import random

import numpy as np
import torch
import torch.distributed as dist

PAB_MANAGER = None


class PABManager:
    def __init__(
        self,
        steps: int = 100,
        cross_broadcast: bool = False,
        cross_threshold: list = [100, 900],
        cross_gap: int = 5,
        spatial_broadcast: bool = False,
        spatial_threshold: list = [100, 900],
        spatial_gap: int = 2,
        temporal_broadcast: bool = False,
        temporal_threshold: list = [100, 900],
        temporal_gap: int = 3,
        diffusion_skip: bool = False,
        diffusion_timestep_respacing: list = None,
        diffusion_skip_timestep: list = None,
        # mlp
        mlp_skip: bool = False,
        mlp_spatial_skip_config: dict = None,
        mlp_temporal_skip_config: dict = None,
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

        self.mlp_skip = mlp_skip
        self.mlp_spatial_skip_config = mlp_spatial_skip_config
        self.mlp_temporal_skip_config = mlp_temporal_skip_config

        self.temporal_mlp_outputs = {}
        self.spatial_mlp_outputs = {}

        if dist.get_rank() == 0:
            print(
                f"\n\
Init SkipManager:\n\
    steps={steps}\n\
    cross_broadcast={cross_broadcast}, cross_threshold={cross_threshold}, cross_gap={cross_gap}\n\
    spatial_broadcast={spatial_broadcast}, spatial_threshold={spatial_threshold}, spatial_gap={spatial_gap}\n\
    temporal_broadcast={temporal_broadcast}, temporal_threshold={temporal_threshold}, temporal_gap={temporal_gap}\n\
    mlp_skip={mlp_skip}\n\
\n",
                end="",
            )

    def if_broadcast_cross(self, timestep: int, count: int):
        if (
            self.cross_broadcast
            and (timestep is not None)
            and (count % self.cross_gap != 0)
            and (self.cross_threshold[0] < timestep < self.cross_threshold[1])
        ):
            flag = True
        else:
            flag = False
        count = (count + 1) % self.steps
        return flag, count

    def if_broadcast_temporal(self, timestep: int, count: int):
        if (
            self.temporal_broadcast
            and (timestep is not None)
            and (count % self.temporal_gap != 0)
            and (self.temporal_threshold[0] < timestep < self.temporal_threshold[1])
        ):
            flag = True
        else:
            flag = False
        count = (count + 1) % self.steps
        return flag, count

    def if_broadcast_spatial(self, timestep: int, count: int, block_idx: int):
        if (
            self.spatial_broadcast
            and (timestep is not None)
            and (count % self.spatial_gap != 0)
            and (self.spatial_threshold[0] < timestep < self.spatial_threshold[1])
        ):
            flag = True
        else:
            flag = False
        count = (count + 1) % self.steps
        return flag, count

    def if_skip_mlp(
        self, timestep: int, count: int, block_idx: int, all_timesteps, is_temporal=False, is_spatial=False
    ):
        # timestep in skip_config? / previous timestep in skip_config?
        def is_t_in_skip_config(all_timesteps, timestep, config):
            is_t_in_skip_config = False
            for key in config:
                index = all_timesteps.index(key)
                skip_range = all_timesteps[index : index + 1 + int(config[key]["skip_count"])]
                if timestep in skip_range:
                    is_t_in_skip_config = True
                    skip_range = [all_timesteps[index], all_timesteps[index + int(config[key]["skip_count"])]]
                    break
            return is_t_in_skip_config, skip_range

        if is_temporal:
            is_t_in_skip_config, skip_range = is_t_in_skip_config(
                all_timesteps, timestep, self.mlp_temporal_skip_config
            )
            next_flag = False
            if (
                self.mlp_skip
                and (timestep is not None)
                and (timestep in self.mlp_temporal_skip_config)
                and (block_idx in self.mlp_temporal_skip_config[timestep]["block"])
            ):
                flag = False
                next_flag = True
                count = count + 1
            elif (
                self.mlp_skip
                and (timestep is not None)
                and (is_t_in_skip_config)
                and (block_idx in self.mlp_temporal_skip_config[skip_range[0]]["block"])
            ):
                flag = True
                count = 0
            else:
                flag = False

            return flag, count, next_flag, skip_range

        elif is_spatial:
            is_t_in_skip_config, skip_range = is_t_in_skip_config(all_timesteps, timestep, self.mlp_spatial_skip_config)
            next_flag = False
            if (
                self.mlp_skip
                and (timestep is not None)
                and (timestep in self.mlp_spatial_skip_config)
                and (block_idx in self.mlp_spatial_skip_config[timestep]["block"])
            ):
                flag = False
                next_flag = True
                count = count + 1
            elif (
                self.mlp_skip
                and (timestep is not None)
                and (is_t_in_skip_config)
                and (block_idx in self.mlp_spatial_skip_config[skip_range[0]]["block"])
            ):
                flag = True
                count = 0
            else:
                flag = False

            return flag, count, next_flag, skip_range

    def save_skip_output(self, timestep, block_idx, ff_output, is_temporal=False, is_spatial=False):
        if is_temporal:
            self.temporal_mlp_outputs[(timestep, block_idx)] = ff_output
        if is_spatial:
            self.spatial_mlp_outputs[(timestep, block_idx)] = ff_output

    def get_skip_output(self, skip_range, timestep, block_idx, is_temporal=False, is_spatial=False):
        skip_start_t = skip_range[0]
        if is_temporal:
            skip_output = (
                self.temporal_mlp_outputs.get((skip_start_t, block_idx), None)
                if self.temporal_mlp_outputs is not None
                else None
            )
        if is_spatial:
            skip_output = (
                self.spatial_mlp_outputs.get((skip_start_t, block_idx), None)
                if self.spatial_mlp_outputs is not None
                else None
            )

        if skip_output is not None:
            if timestep == skip_range[-1]:
                print(f"skip_range | [{skip_range[0]}, {skip_range[-1]}]")
                # TODO save memory
                if is_temporal:
                    del self.temporal_mlp_outputs[(skip_start_t, block_idx)]
                if is_spatial:
                    del self.spatial_mlp_outputs[(skip_start_t, block_idx)]
        else:
            raise ValueError(
                f"No stored MLP output found | t {timestep} |[{skip_range[0]}, {skip_range[-1]}] | block {block_idx}"
            )

        # if skip_output is not None:
        #     print(f"skip_range | [{skip_range[0]}, {skip_range[-1]}]")
        #     if is_temporal:
        #         print(
        #             f"Skip | Using stored MLP output | Time | t {timestep} | [{skip_range[0]}, {skip_range[-1]}] | block {block_idx}"
        #         )
        #     else:
        #         print(
        #             f"Skip | Using stored MLP output | Spatial | t {timestep} | [{skip_range[0]}, {skip_range[-1]}] | block {block_idx}"
        #         )

        #     if timestep == skip_range[-1]:
        #         if is_temporal:
        #             del self.temporal_mlp_outputs[(skip_start_t, block_idx)]
        #         if is_spatial:
        #             del self.spatial_mlp_outputs[(skip_start_t, block_idx)]
        #         print(
        #             f"Skip | Delete stored MLP output | t {timestep} | [{skip_range[0]}, {skip_range[-1]}] | block {block_idx}"
        #         )
        # else:
        #     raise ValueError(
        #         f"No stored MLP output found | t {timestep} |[{skip_range[0]}, {skip_range[-1]}] | block {block_idx}"
        #     )

        return skip_output

    def get_spatial_mlp_outputs(self):
        return self.spatial_mlp_outputs

    def get_temporal_mlp_outputs(self):
        return self.temporal_mlp_outputs


def set_pab_manager(
    steps: int = 100,
    cross_broadcast: bool = False,
    cross_threshold: list = [100, 900],
    cross_gap: int = 5,
    spatial_broadcast: bool = False,
    spatial_threshold: list = [100, 900],
    spatial_gap: int = 2,
    temporal_broadcast: bool = False,
    temporal_threshold: list = [100, 900],
    temporal_gap: int = 3,
    diffusion_skip: bool = False,
    diffusion_timestep_respacing: list = None,
    diffusion_skip_timestep: list = None,
    # mlp
    mlp_skip: bool = False,
    mlp_spatial_skip_config: dict = None,
    mlp_temporal_skip_config: dict = None,
):
    global PAB_MANAGER
    PAB_MANAGER = PABManager(
        steps,
        cross_broadcast,
        cross_threshold,
        cross_gap,
        spatial_broadcast,
        spatial_threshold,
        spatial_gap,
        temporal_broadcast,
        temporal_threshold,
        temporal_gap,
        diffusion_skip,
        diffusion_timestep_respacing,
        diffusion_skip_timestep,
        mlp_skip,
        mlp_spatial_skip_config,
        mlp_temporal_skip_config,
    )


def enable_pab():
    if PAB_MANAGER is None:
        return False
    return PAB_MANAGER.cross_broadcast or PAB_MANAGER.spatial_broadcast or PAB_MANAGER.temporal_broadcast


def if_broadcast_cross(timestep: int, count: int):
    if not enable_pab():
        return False, count
    return PAB_MANAGER.if_broadcast_cross(timestep, count)


def if_broadcast_temporal(timestep: int, count: int):
    if not enable_pab():
        return False, count
    return PAB_MANAGER.if_broadcast_temporal(timestep, count)


def if_broadcast_spatial(timestep: int, count: int, block_idx: int):
    if not enable_pab():
        return False, count
    return PAB_MANAGER.if_broadcast_spatial(timestep, count, block_idx)


def if_skip_mlp(timestep: int, count: int, block_idx: int, all_timesteps, is_temporal=False, is_spatial=False):
    return PAB_MANAGER.if_skip_mlp(timestep, count, block_idx, all_timesteps, is_temporal, is_spatial)


def save_skip_output(timestep: int, block_idx: int, ff_output, is_temporal=False, is_spatial=False):
    return PAB_MANAGER.save_skip_output(timestep, block_idx, ff_output, is_temporal, is_spatial)


def get_skip_output(skip_range, timestep, block_idx: int, is_temporal=False, is_spatial=False):
    return PAB_MANAGER.get_skip_output(skip_range, timestep, block_idx, is_temporal, is_spatial)


def get_diffusion_skip():
    return enable_pab() and PAB_MANAGER.diffusion_skip


def get_diffusion_timestep_respacing():
    return PAB_MANAGER.diffusion_timestep_respacing


def get_diffusion_skip_timestep():
    return enable_pab() and PAB_MANAGER.diffusion_skip_timestep


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