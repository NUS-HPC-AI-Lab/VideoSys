import random
from typing import Optional

import numpy as np
import torch
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _get_default_group
from torch.utils.data import DataLoader

from .datasets import DummyVariableVideoTextDataset, VariableVideoTextDataset, VideoTextDataset
from .sampler import StatefulDistributedSampler, VariableVideoBatchSampler


# Deterministic dataloader
def get_seed_worker(seed):
    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


def prepare_dataloader(
    dataset,
    batch_size: Optional[int] = None,
    shuffle: bool = False,
    seed: int = 1024,
    drop_last: bool = False,
    pin_memory: bool = True,
    num_workers: int = 0,
    process_group: Optional[ProcessGroup] = None,
    bucket_config=None,
    num_bucket_build_workers: int = 1,
    prefetch_factor: Optional[int] = None,
    optimized_schedule: Optional[str] = None,
    sp_balance_scope: str = "iter",
    auto_grad_accumulation: bool = False,
    max_grad_accumulation_steps: int = 2,
    parallel_mgr=None,
    **kwargs,
):
    _kwargs = kwargs.copy()
    if isinstance(dataset, (VariableVideoTextDataset, DummyVariableVideoTextDataset)):
        batch_sampler = VariableVideoBatchSampler(
            dataset,
            bucket_config,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last,
            verbose=True,
            num_bucket_build_workers=num_bucket_build_workers,
            optimized_schedule=optimized_schedule,
            sp_balance_scope=sp_balance_scope,
            auto_grad_accumulation=auto_grad_accumulation,
            max_grad_accumulation_steps=max_grad_accumulation_steps,
            parallel_mgr=parallel_mgr,
        )
        return (
            DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                worker_init_fn=get_seed_worker(seed),
                pin_memory=pin_memory,
                num_workers=num_workers,
                collate_fn=_collate_fn,
                prefetch_factor=prefetch_factor,
                **_kwargs,
            ),
            batch_sampler,
        )
    elif isinstance(dataset, VideoTextDataset):
        process_group = process_group or _get_default_group()
        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
            shuffle=shuffle,
        )
        return (
            DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                worker_init_fn=get_seed_worker(seed),
                drop_last=drop_last,
                pin_memory=pin_memory,
                num_workers=num_workers,
                collate_fn=collate_fn_default,
                prefetch_factor=prefetch_factor,
                **_kwargs,
            ),
            sampler,
        )
    else:
        raise ValueError(f"Unsupported dataset type: {type(dataset)}")


def _collate_fn(batch):
    ar_name = batch[0]["ar_name"]
    num_frame = batch[0]["num_frames"]
    sp_size = batch[0]["sp_size"]
    gas = batch[0]["gas"]

    stride = (len(batch) + gas - 1) // gas
    ret = dict(ar_name=ar_name, num_frame=num_frame, sp_size=sp_size, gas=gas, data=[])
    for i in range(0, len(batch), stride):
        assert all(each.pop("sp_size") == sp_size for each in batch[i : i + stride])
        assert all(each.pop("gas") == gas for each in batch[i : i + stride])
        assert all(each.pop("ar_name") == ar_name for each in batch[i : i + stride])
        assert all(each["num_frames"] == num_frame for each in batch[i : i + stride])

        ret["data"].append(torch.utils.data.default_collate(batch[i : i + stride]))
    return ret


def collate_fn_default(batch):
    # HACK: for loading text features
    use_mask = False
    if "mask" in batch[0] and isinstance(batch[0]["mask"], int):
        masks = [x.pop("mask") for x in batch]

        texts = [x.pop("text") for x in batch]
        texts = torch.cat(texts, dim=1)
        use_mask = True

    ret = torch.utils.data.default_collate(batch)

    if use_mask:
        ret["mask"] = masks
        ret["text"] = texts
    return ret
