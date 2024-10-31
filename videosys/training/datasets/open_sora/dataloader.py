import collections
import random
from typing import Optional

import numpy as np
import torch
from torch.distributed import ProcessGroup
from torch.distributed.distributed_c10d import _get_default_group
from torch.utils.data import DataLoader

from .datasets import BatchFeatureDataset, DummyVariableVideoTextDataset, VariableVideoTextDataset, VideoTextDataset
from .sampler import BatchDistributedSampler, StatefulDistributedSampler, VariableVideoBatchSampler


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
    batch_size=None,
    shuffle=False,
    seed=1024,
    drop_last=False,
    pin_memory=False,
    num_workers=0,
    process_group: Optional[ProcessGroup] = None,
    bucket_config=None,
    num_bucket_build_workers=1,
    prefetch_factor=None,
    optimized_schedule: str = None,
    auto_grad_accumulation: bool = False,
    keep_last: bool = False,
    max_grad_accumulation_steps: int = 2,
    preprocessed_data=True,
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
            keep_last=keep_last,
            verbose=True,
            num_bucket_build_workers=num_bucket_build_workers,
            optimized_schedule=optimized_schedule,
            auto_grad_accumulation=auto_grad_accumulation,
            max_grad_accumulation_steps=max_grad_accumulation_steps,
        )
        return (
            DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                worker_init_fn=get_seed_worker(seed),
                pin_memory=pin_memory,
                num_workers=num_workers,
                collate_fn=_collate_fn if preprocessed_data else collate_fn_default,
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
    elif isinstance(dataset, BatchFeatureDataset):
        sampler = BatchDistributedSampler(
            dataset,
            num_replicas=process_group.size(),
            rank=process_group.rank(),
        )
        return (
            DataLoader(
                dataset,
                batch_size=1,
                sampler=sampler,
                worker_init_fn=get_seed_worker(seed),
                pin_memory=pin_memory,
                num_workers=num_workers,
                collate_fn=collate_fn_batch,
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
        # print(f"{[(x['height'], x['width'], x['num_frames'], x['video'].shape, x['id']) for x in batch[i:i+gas]]}")
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


def collate_fn_batch(batch):
    """
    Used only with BatchDistributedSampler
    """
    res = torch.utils.data.default_collate(batch)

    # squeeze the first dimension, which is due to torch.stack() in default_collate()
    if isinstance(res, collections.abc.Mapping):
        for k, v in res.items():
            if isinstance(v, torch.Tensor):
                res[k] = v.squeeze(0)
    elif isinstance(res, collections.abc.Sequence):
        res = [x.squeeze(0) if isinstance(x, torch.Tensor) else x for x in res]
    elif isinstance(res, torch.Tensor):
        res = res.squeeze(0)
    else:
        raise TypeError

    return res
