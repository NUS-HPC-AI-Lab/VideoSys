from functools import partial
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from videosys.core.comm import all_reduce, all_to_all_with_pad, gather_sequence, halo_exchange, split_sequence
from videosys.core.parallel_mgr import (
    get_sequence_parallel_group,
    get_sequence_parallel_rank,
    get_sequence_parallel_size,
)


def _replace_stres_fwd(module: torch.nn.Module):
    bound_method = _forward_stres.__get__(module, module.__class__)
    setattr(module, "forward", bound_method)
    if module.temporal_res_block.conv1.kernel_size[1] != 0:
        bound_method = _forward_conv3d_sp.__get__(module, module.__class__)
        _replace_conv_fwd(module.temporal_res_block.conv1)
        _replace_conv_fwd(module.temporal_res_block.conv2)
        if module.temporal_res_block.use_in_shortcut:
            _replace_conv_fwd(module.temporal_res_block.conv_shortcut)
    _replace_groupnorm_fwd(module.temporal_res_block.norm1)
    _replace_groupnorm_fwd(module.temporal_res_block.norm2)


def _replace_mid_fwd(module: torch.nn.Module):
    bound_method = _forward_mid.__get__(module, module.__class__)
    setattr(module, "forward", bound_method)


def _replace_up_fwd(module: torch.nn.Module):
    bound_method = _forward_up.__get__(module, module.__class__)
    setattr(module, "forward", bound_method)


def _replace_decoder_fwd(module: torch.nn.Module):
    bound_method = _forward_decoder.__get__(module, module.__class__)
    setattr(module, "forward", bound_method)


def _replace_groupnorm_fwd(module: nn.GroupNorm):
    bound_method = dist_groupnorm.__get__(module, module.__class__)
    bound_method = partial(
        bound_method,
        weight=module.weight,
        bias=module.bias,
        eps=module.eps,
        group_num=module.num_groups,
        group=get_sequence_parallel_group(),
    )
    setattr(module, "forward", bound_method)


def _replace_conv_fwd(module: nn.Conv3d):
    bound_method = _forward_conv3d_sp.__get__(module, module.__class__)
    setattr(module, "forward", bound_method)
    set_sp_padding(module)

def _replace_conv_opensora_fwd(module: nn.Module):
    bound_method = _forward_conv3d_opensora.__get__(module, module.__class__)
    setattr(module, "forward", bound_method)


def dist_groupnorm(
    self, x: torch.Tensor, group_num: int, weight: torch.Tensor, bias: torch.Tensor, eps: float, group
) -> torch.Tensor:
    # x: input features with shape [N,C,H,W]
    # weight, bias: scale and offset, with shape [C]
    # group_num: number of groups for GN

    x_shape = x.shape
    batch_size = x_shape[0]
    dtype = x.dtype
    x = x.to(torch.float32)
    x = x.reshape(batch_size, group_num, -1)

    mean = x.mean(dim=-1, keepdim=True)
    mean = all_reduce(mean, group)
    mean = mean / dist.get_world_size(group)

    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    var = all_reduce(var, group)
    var = var / dist.get_world_size(group)

    x = (x - mean) / torch.sqrt(var + eps)

    x = x.view(x_shape).to(dtype)
    x = weight.view(1, -1, 1, 1, 1) * x + bias.view(1, -1, 1, 1, 1)
    return x


def set_sp_padding(module: nn.Conv3d):
    padding_ = module.padding
    padding = []
    for i in range(3):
        padding.insert(0, padding_[i])
        padding.insert(0, padding_[i])
    module.width_pad = padding[0]
    if get_sequence_parallel_rank() == 0:
        padding[1] = 0
    elif get_sequence_parallel_rank() == get_sequence_parallel_size() - 1:
        padding[0] = 0
    else:
        padding[0] = 0
        padding[1] = 0
    module.padding = (0, 0, 0)
    module.padding_mode = "zeros"
    module._reversed_padding_repeated_twice = tuple(padding)


def _forward_conv3d_sp(self, x):
    halo_size = self.kernel_size[2] // 2
    x = halo_exchange(x, get_sequence_parallel_group(), 4, halo_size)
    x = F.pad(x, self._reversed_padding_repeated_twice, mode="constant", value=0)
    ret = self._conv_forward(x, self.weight, self.bias)
    return ret


def dynamic_switch(x, to_spatial_shard: bool, scatter_dim: int = 0, gather_dim: int = 3):
    if to_spatial_shard:
        scatter_dim, gather_dim = gather_dim, scatter_dim

    if x.shape[scatter_dim] % get_sequence_parallel_size() != 0 or x.shape[gather_dim] < get_sequence_parallel_size():
        return x
    x = all_to_all_with_pad(
        x,
        get_sequence_parallel_group(),
        scatter_dim=scatter_dim,
        gather_dim=gather_dim,
    )
    return x


def _forward_stres(
    self,
    hidden_states: torch.Tensor,
    temb: Optional[torch.Tensor] = None,
    image_only_indicator: Optional[torch.Tensor] = None,
):
    hidden_states = dynamic_switch(hidden_states, to_spatial_shard=False)
    num_frames = image_only_indicator.shape[-1]
    hidden_states = self.spatial_res_block(hidden_states, temb)
    hidden_states = dynamic_switch(hidden_states, to_spatial_shard=True)

    batch_frames, channels, height, width = hidden_states.shape
    batch_size = batch_frames // num_frames

    hidden_states_mix = (
        hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
    )
    hidden_states = (
        hidden_states[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
    )

    if temb is not None:
        temb = temb.reshape(batch_size, num_frames, -1)

    hidden_states = self.temporal_res_block(hidden_states, temb)
    hidden_states = self.time_mixer(
        x_spatial=hidden_states_mix,
        x_temporal=hidden_states,
        image_only_indicator=image_only_indicator,
    )

    hidden_states = hidden_states.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)
    return hidden_states


def _forward_mid(
    self,
    hidden_states: torch.Tensor,
    image_only_indicator: torch.Tensor,
):
    hidden_states = self.resnets[0](
        hidden_states,
        image_only_indicator=image_only_indicator,
    )
    for resnet, attn in zip(self.resnets[1:], self.attentions):
        hidden_states = dynamic_switch(hidden_states, to_spatial_shard=False)
        hidden_states = attn(hidden_states)
        hidden_states = dynamic_switch(hidden_states, to_spatial_shard=True)
        hidden_states = resnet(
            hidden_states,
            image_only_indicator=image_only_indicator,
        )
    return hidden_states


def _forward_up(
    self,
    hidden_states: torch.Tensor,
    image_only_indicator: torch.Tensor,
) -> torch.Tensor:
    for resnet in self.resnets:
        hidden_states = resnet(
            hidden_states,
            image_only_indicator=image_only_indicator,
        )

    if self.upsamplers is not None:
        hidden_states = dynamic_switch(hidden_states, to_spatial_shard=False)
        for upsampler in self.upsamplers:
            hidden_states = upsampler(hidden_states)
        hidden_states = dynamic_switch(hidden_states, to_spatial_shard=True)

    return hidden_states


def _forward_decoder(
    self,
    sample: torch.Tensor,
    image_only_indicator: torch.Tensor,
    num_frames: int = 1,
) -> torch.Tensor:
    r"""The forward method of the `Decoder` class."""

    sample = self.conv_in(sample)

    upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

    # middle
    padding_s = sample.shape[-1] % get_sequence_parallel_size()
    padding_f = sample.shape[0] % get_sequence_parallel_size()
    sample = F.pad(sample, (0, padding_s, 0, 0, 0, padding_f))
    sample = split_sequence(sample, get_sequence_parallel_group(), dim=3)
    sample = self.mid_block(sample, image_only_indicator=image_only_indicator)
    sample = sample.to(upscale_dtype)

    # up
    for up_block in self.up_blocks:
        sample = up_block(sample, image_only_indicator=image_only_indicator)

    sample = gather_sequence(sample, get_sequence_parallel_group(), dim=3)
    sample = sample.narrow(3, 0, sample.shape[3] - padding_s)
    sample = sample.narrow(0, 0, sample.shape[0] - padding_f)
    # post-process
    sample = self.conv_norm_out(sample)
    sample = self.conv_act(sample)
    sample = self.conv_out(sample)

    batch_frames, channels, height, width = sample.shape
    batch_size = batch_frames // num_frames
    sample = sample[None, :].reshape(batch_size, num_frames, channels, height, width).permute(0, 2, 1, 3, 4)
    sample = self.time_conv_out(sample)

    sample = sample.permute(0, 2, 1, 3, 4).reshape(batch_frames, channels, height, width)
    return sample

def _forward_conv3d_opensora(self, x):
    time_causal_padding = list(self.time_causal_padding)
    if get_sequence_parallel_rank() == 0:
        time_causal_padding[1] = 0
    elif get_sequence_parallel_rank() == get_sequence_parallel_size() - 1:
        time_causal_padding[0] = 0
    else:
        time_causal_padding[0] = 0
        time_causal_padding[1] = 0
    time_causal_padding = tuple(time_causal_padding)
    x = F.pad(x, time_causal_padding, mode=self.pad_mode)
    x = self.conv(x)
    return x