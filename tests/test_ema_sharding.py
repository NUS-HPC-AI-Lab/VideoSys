from copy import deepcopy

import colossalai
import torch
import torch.distributed as dist
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from colossalai.nn.optimizer import HybridAdam
from colossalai.testing import clear_cache_before_run, rerun_if_address_is_in_use, spawn

from opendit.models.dit import DiT_S_2
from opendit.utils.ckpt_utils import model_gathering, record_model_param_shape
from opendit.utils.operation import model_sharding
from opendit.utils.train_utils import update_ema


@clear_cache_before_run()
def run_ema_sharding():
    plugin = LowLevelZeroPlugin(precision="fp16", stage=2, max_norm=1.0, initial_scale=32)
    booster = Booster(plugin=plugin)
    model = DiT_S_2().cuda().half()

    ema_sharding = deepcopy(model).eval()
    model_param_shape = record_model_param_shape(ema_sharding)
    model_sharding(ema_sharding)
    ema_no_sharding = deepcopy(model).eval()

    criterion = lambda x: x.mean()
    optimizer = HybridAdam((model.parameters()), lr=0.001)
    model, optimizer, criterion, _, _ = booster.boost(model, optimizer, criterion)

    x = torch.randn(2, 4, 32, 32).cuda().requires_grad_(True)
    y = torch.randint(0, 10, (2,)).cuda()
    t = torch.randint(0, 10, (2,)).cuda()
    output = model(x, y, t)
    loss = criterion(output)
    booster.backward(loss, optimizer)
    optimizer.step()

    update_ema(ema_sharding, model.module, optimizer=optimizer, sharded=True, decay=0.5)
    update_ema(ema_no_sharding, model.module, optimizer=optimizer, sharded=False, decay=0.5)

    # should be equal after update
    gather_ema_sharding = deepcopy(ema_sharding)
    model_gathering(gather_ema_sharding, model_param_shape)
    if dist.get_rank() == 0:
        for (gather_ema_sharding_name, gather_ema_sharding_param), (ema_no_sharding_name, ema_no_sharding_param) in zip(
            gather_ema_sharding.named_parameters(), ema_no_sharding.named_parameters()
        ):
            assert gather_ema_sharding_name == ema_no_sharding_name
            if gather_ema_sharding_name == "pos_embed":
                continue
            assert torch.allclose(gather_ema_sharding_param, ema_no_sharding_param)
    dist.barrier()

    # should be same after sharding again
    if dist.get_rank() == 0:
        model_sharding(gather_ema_sharding)
        for (gather_ema_sharding_name, gather_ema_sharding_param), (ema_sharding_name, ema_sharding_param) in zip(
            gather_ema_sharding.named_parameters(), ema_sharding.named_parameters()
        ):
            assert gather_ema_sharding_name == ema_sharding_name
            if gather_ema_sharding_name == "pos_embed":
                continue
            assert torch.allclose(gather_ema_sharding_param, ema_sharding_param)
    dist.barrier()


def run_dist(rank, world_size, port):
    colossalai.launch(config=(dict()), rank=rank, world_size=world_size, port=port, host="localhost")
    run_ema_sharding()
    torch.cuda.empty_cache()


@rerun_if_address_is_in_use()
def test_ema_sharding():
    spawn(run_dist, 2)


if __name__ == "__main__":
    test_ema_sharding()
