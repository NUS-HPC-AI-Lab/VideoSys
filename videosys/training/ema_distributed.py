from collections import OrderedDict

import torch
import torch.distributed as dist


@torch.no_grad()
def update_ema(
    ema_model: torch.nn.Module, model: torch.nn.Module, optimizer=None, decay: float = 0.9999, sharded: bool = True
) -> None:
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # skip parameters that are not trainable
        if name == "pos_embed" or param.requires_grad == False:
            continue
        # update non-sharded ema
        if not sharded:
            param_data = param.data
            ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)
        # update sharded ema
        else:
            if param._hp_mapping is not None:
                param_data = param._hp_mapping.get_hp_fragment()
                param_data = param_data.reshape_as(ema_params[name])
                ema_params[name].mul_(decay).add_(param_data, alpha=1 - decay)


def ema_sharding(model: torch.nn.Module, ema_model: torch.nn.Module) -> None:
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name == "pos_embed" or param.requires_grad == False:
            continue
        if param._hp_mapping is not None:
            # has param on this rank
            param_data = param._hp_mapping.get_hp_fragment()
        else:
            # no param on this rank
            param_data = torch.empty((1,)).to(param.device)
        ema_params[name].data = param_data


def ema_gathering(model: torch.nn.Module, ema_model: torch.nn.Module):
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        if name == "pos_embed" or param.requires_grad == False:
            continue
        reduce_buffer = torch.zeros_like(param, dtype=torch.float32).flatten()
        if param._hp_mapping is not None:
            ema_param = ema_params[name]
            lp_frag_address = param._hp_mapping.lp_fragment_address
            reduce_fragment = torch.narrow(reduce_buffer, 0, lp_frag_address.start, lp_frag_address.numel)
            reduce_fragment.data.copy_(ema_param.data)
        dist.all_reduce(reduce_buffer, group=param._dp_group)
        ema_params[name].data = reduce_buffer.reshape_as(param)
