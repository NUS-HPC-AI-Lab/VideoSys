import json
import os
from collections import OrderedDict, defaultdict
from functools import partial
from pprint import pformat
from typing import Iterator, List, Optional, Union

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DistributedSampler

from videosys.core.profiler import get_profiler
from videosys.core.recompute import STDiT3BlockRecomputeConfig, disable_profile, enable_profile, get_profile_context
from videosys.utils.logging import logger
from videosys.utils.training import GroupTimer

from .aspect import DEFAULT_AR_MAP
from .bucket import Bucket
from .datasets import DummyVariableVideoTextDataset, VariableVideoTextDataset

GB = 1024**3


# use pandarallel to accelerate bucket processing
# NOTE: pandarallel should only access local variables
def apply(data, method=None, frame_interval=None, seed=None, num_bucket=None):
    return method(
        data["num_frames"],
        data["height"],
        data["width"],
        frame_interval,
        seed + data["id"] * num_bucket,
    )


def print_memory_stats(phase=None):
    free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
    alloc_mem, resrv_mem, max_alloc, max_resrv = (
        torch.cuda.memory_allocated(),
        torch.cuda.memory_reserved(),
        torch.cuda.max_memory_allocated(),
        torch.cuda.max_memory_reserved(),
    )
    if phase is not None:
        phase = f"after {phase}"
    else:
        phase = ""
    print(
        f">>> [Profiling {dist.get_rank()} {phase}] free GPU memory: {free_gpu_memory/GB:.2f}/{total_gpu_memory/GB:.2f}, "
        f"allocated memory: {alloc_mem/GB:.2f} GB, reserved memory: {resrv_mem/GB:.2f} GB, "
        f"max allocated memory: {max_alloc/GB:.2f} GB, max reserved memory: {max_resrv/GB:.2f} GB"
    )
    return free_gpu_memory, total_gpu_memory, alloc_mem, resrv_mem, max_alloc, max_resrv


def reset_status(model, optimizer):
    # reset model status
    model.zero_grad()

    # reset optimizer status
    if optimizer.contiguous_gradients:
        optimizer.ipg_buffer = None
        optimizer.grads_in_partition = None
        optimizer.grads_in_partition_offset = 0
    for i in range(len(optimizer.params_already_reduced)):
        optimizer.params_already_reduced[i] = False
    optimizer.grads_in_ipg_bucket = []
    optimizer.params_in_ipg_bucket = []
    optimizer.ipg_bucket_has_moe_params = False
    optimizer.elements_in_ipg_bucket = 0
    optimizer.extra_large_param_to_reduce = None
    optimizer.zero_grad()
    clean_cache()


def clean_cache():
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


class StatefulDistributedSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.start_index: int = 0

    def __iter__(self) -> Iterator:
        iterator = super().__iter__()
        indices = list(iterator)
        indices = indices[self.start_index :]
        return iter(indices)

    def __len__(self) -> int:
        return self.num_samples - self.start_index

    def reset(self) -> None:
        self.start_index = 0

    def state_dict(self, step) -> dict:
        return {"start_index": step}

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)


class VariableVideoBatchSampler(DistributedSampler):
    def __init__(
        self,
        dataset: Union[VariableVideoTextDataset, DummyVariableVideoTextDataset],
        bucket_config: dict,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        keep_last: bool = False,
        verbose: bool = False,
        num_bucket_build_workers: int = 1,
        optimized_schedule: str = None,
        auto_grad_accumulation: bool = False,
        max_grad_accumulation_steps: int = 2,
        parallel_mgr=None,
    ) -> None:
        super().__init__(
            dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last
        )
        self.dataset = dataset
        self.bucket = Bucket(bucket_config)
        self.verbose = verbose
        self.last_micro_batch_access_index = 0
        self.approximate_num_batch = None
        self.keep_last = keep_last

        self._get_num_batch_cached_bucket_sample_dict = None
        self.num_bucket_build_workers = num_bucket_build_workers

        self.optimized_schedule = optimized_schedule
        self.auto_grad_accumulation = auto_grad_accumulation
        self.max_grad_accumulation_steps = max_grad_accumulation_steps
        self.profiler = get_profiler()
        self.generator = None
        if self.shuffle:
            self.generator = torch.Generator()
            self.generator.manual_seed(self.seed + self.epoch)
        self.cached_bucket_id_access_order = None
        self.effective_samples = 0
        self.parallel_mgr = parallel_mgr

    def __iter__(self) -> Iterator[List[int]]:
        if self._get_num_batch_cached_bucket_sample_dict is not None:
            bucket_sample_dict = self._get_num_batch_cached_bucket_sample_dict
            self._get_num_batch_cached_bucket_sample_dict = None
        else:
            bucket_sample_dict = self.group_by_bucket()
            if self.optimized_schedule is not None:
                self.get_num_batch_with_optimized_schedule(bucket_sample_dict, self.optimized_schedule == "global")
            else:
                self.get_num_batch(bucket_sample_dict)

        if self.optimized_schedule is not None:
            yield from self._optimized_schedule_iter(bucket_sample_dict, self.optimized_schedule == "global")
        else:
            yield from self._bucketized_iter(bucket_sample_dict)

    def change_timer_group(self, timers):
        cur_group = self.parallel_mgr.sp_group
        for t in timers:
            timers[t].group = cur_group

    def _build_bucketized_bucket_id_access_order(self, bucket_sample_dict):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        bucket_micro_batch_count = OrderedDict()
        self.effective_samples = 0

        # process the samples
        for bucket_id, data_list in bucket_sample_dict.items():
            ar_name, num_frame = bucket_id[:2]
            if not self.profiler.is_valid_bucket(ar_name, num_frame):
                logger.info(f"skip building batches for bucket {bucket_id} because it's invalid")
                continue
            # handle droplast
            bs_per_gpu = self.get_batch_size(bucket_id)
            org_num_samples = len(data_list)
            remainder = org_num_samples % bs_per_gpu

            if (not self.keep_last) and remainder > 0:
                if not self.drop_last:
                    # if there is remainder, we pad to make it divisible
                    pad = bs_per_gpu - remainder
                    if pad > org_num_samples:
                        data_list = data_list * ((pad + org_num_samples - 1) // org_num_samples + 1)
                        data_list = data_list[: pad + org_num_samples]
                    else:
                        data_list += data_list[:pad]
                else:
                    # we just drop the remainder to make it divisible
                    data_list = data_list[:-remainder]
            bucket_sample_dict[bucket_id] = data_list

            # handle shuffle
            if self.shuffle:
                data_indices = torch.randperm(len(data_list), generator=g).tolist()
                data_list = [data_list[i] for i in data_indices]
                bucket_sample_dict[bucket_id] = data_list

            # compute how many micro-batches each bucket has
            effect_size = len(data_list)
            if self.keep_last or (not self.drop_last):
                num_micro_batches = (effect_size + bs_per_gpu - 1) // bs_per_gpu
            else:
                num_micro_batches = effect_size // bs_per_gpu
            bucket_micro_batch_count[bucket_id] = num_micro_batches
            if not self.keep_last:
                self.effective_samples += bs_per_gpu * num_micro_batches
            else:
                self.effective_samples += effect_size

        # compute the bucket access order
        # each bucket may have more than one batch of data
        # thus bucket_id may appear more than 1 time
        bucket_id_access_order = []
        for bucket_id, num_micro_batch in bucket_micro_batch_count.items():
            bucket_id_access_order.extend([bucket_id] * num_micro_batch)

        # randomize the access order
        if self.shuffle:
            bucket_id_access_order_indices = torch.randperm(len(bucket_id_access_order), generator=g).tolist()
            bucket_id_access_order = [bucket_id_access_order[i] for i in bucket_id_access_order_indices]

        # make the number of bucket accesses divisible by dp size
        original_batches = len(bucket_id_access_order)
        remainder = original_batches % self.num_replicas
        bucket_num_batch_to_deduct = defaultdict(int)
        if remainder > 0:
            # bucket_num_samples_to_deduct = defaultdict(int)
            for i in range(original_batches - remainder, original_batches):
                bucket_num_batch_to_deduct[bucket_id_access_order[i]] += 1

            bucket_id_access_order = bucket_id_access_order[: original_batches - remainder]

            for bucket_id, num_batch_to_deduct in bucket_num_batch_to_deduct.items():
                total_samples = len(bucket_sample_dict[bucket_id])
                total_batches = bucket_micro_batch_count[bucket_id]

                left_batchs = total_batches - num_batch_to_deduct
                left_samples = left_batchs * self.get_batch_size(bucket_id)
                self.effective_samples -= total_samples - left_samples

        for i in range(len(bucket_id_access_order)):
            logger.info(f"iter {i}, bucket_id: {bucket_id_access_order[i]}")
        logger.info(f"dropped: {pformat(bucket_num_batch_to_deduct, sort_dicts=False)}")
        return bucket_id_access_order

    def _bucketized_iter(self, bucket_sample_dict):
        bucket_last_consumed = OrderedDict()
        # acc_num_samples = torch.zeros(1, device=torch.cuda.current_device(), dtype=torch.float)
        if self.cached_bucket_id_access_order is not None:
            bucket_id_access_order = self.cached_bucket_id_access_order
            self.cached_bucket_id_access_order = None
        else:
            bucket_id_access_order = self._build_bucketized_bucket_id_access_order(bucket_sample_dict)

        # prepare each batch from its bucket
        # according to the predefined bucket access order
        num_iters = len(bucket_id_access_order) // self.num_replicas
        start_iter_idx = self.last_micro_batch_access_index // self.num_replicas

        # re-compute the micro-batch consumption
        # this is useful when resuming from a state dict with a different number of GPUs
        self.last_micro_batch_access_index = start_iter_idx * self.num_replicas
        for i in range(self.last_micro_batch_access_index):
            bucket_id = bucket_id_access_order[i]
            bucket_bs = self.get_batch_size(bucket_id)
            if bucket_id in bucket_last_consumed:
                bucket_last_consumed[bucket_id] += bucket_bs
            else:
                bucket_last_consumed[bucket_id] = bucket_bs

        for i in range(start_iter_idx, num_iters):
            bucket_access_list = bucket_id_access_order[i * self.num_replicas : (i + 1) * self.num_replicas]
            self.last_micro_batch_access_index += self.num_replicas

            # compute the data samples consumed by each access
            bucket_access_boundaries = []
            for bucket_id in bucket_access_list:
                bucket_bs = self.get_batch_size(bucket_id)
                last_consumed_index = bucket_last_consumed.get(bucket_id, 0)
                bucket_access_boundaries.append([last_consumed_index, last_consumed_index + bucket_bs])

                # update consumption
                if bucket_id in bucket_last_consumed:
                    bucket_last_consumed[bucket_id] += bucket_bs
                else:
                    bucket_last_consumed[bucket_id] = bucket_bs

            # compute the range of data accessed by each GPU
            bucket_id = bucket_access_list[self.rank]
            boundary = bucket_access_boundaries[self.rank]
            cur_micro_batch = bucket_sample_dict[bucket_id][boundary[0] : boundary[1]]

            # DEBUG purpose
            # print(f"rank: {self.rank}, iter {i} bucket_id: {bucket_id}, len: {len(cur_micro_batch)}, expect bs: {self.get_batch_size(bucket_id)}, consumed: {boundary}, len: {len(bucket_sample_dict[bucket_id])}")
            # acc_num_samples += len(cur_micro_batch)
            # assert len(cur_micro_batch) == self.get_batch_size(
            #     bucket_id
            # ), f"rank: {self.rank}, iter {i} bucket_id: {bucket_id}, len: {len(cur_micro_batch)}, expect bs: {self.get_batch_size(bucket_id)}, consumed: {boundary}, len: {len(bucket_sample_dict[bucket_id])}"
            # encode t, h, w into the sample index
            real_t, real_h, real_w = self.bucket.get_thw(bucket_id)
            cur_micro_batch = [
                (idx, real_t, real_h, real_w, bucket_id[0], self.parallel_mgr.sp_size, 1) for idx in cur_micro_batch
            ]
            yield cur_micro_batch

        # print(f"rank: {self.rank}, acc_num_samples: {acc_num_samples}")
        self.reset()

    def get_batch_size(self, bucket_id):
        bs_from_bucket_config = self.profiler.get_batch_size(bucket_id[0], bucket_id[1])
        return bs_from_bucket_config

    def __len__(self) -> int:
        bucket_sample_dict = self.group_by_bucket()
        self._get_num_batch_cached_bucket_sample_dict = bucket_sample_dict

        if self.optimized_schedule is not None:
            return self.get_num_batch_with_optimized_schedule(bucket_sample_dict, self.optimized_schedule == "global")
        else:
            return self.get_num_batch(bucket_sample_dict) // self.num_replicas

    def group_by_bucket(self) -> dict:
        bucket_sample_dict = OrderedDict()

        from pandarallel import pandarallel

        pandarallel.initialize(nb_workers=self.num_bucket_build_workers, progress_bar=False)
        logger.info(f"Building buckets...")
        bucket_ids = self.dataset.data.parallel_apply(
            apply,
            axis=1,
            method=self.bucket.get_bucket_id,
            frame_interval=self.dataset.frame_interval,
            seed=self.seed + self.epoch,
            num_bucket=self.bucket.num_bucket,
        )

        # group by bucket
        # each data sample is put into a bucket with a similar image/video size
        for i in range(len(self.dataset)):
            bucket_id = bucket_ids[i]
            if bucket_id is None:
                continue
            if bucket_id not in bucket_sample_dict:
                bucket_sample_dict[bucket_id] = []
            bucket_sample_dict[bucket_id].append(i)
        return bucket_sample_dict

    def _group_bucket_by_sp_size(self, bucket_sample_dict):
        # group the buckets by sp size, and collect the micro batch access order for each sp size
        self.effective_samples = 0
        sp_bucket_id_access_order = defaultdict(list)
        for bucket_id, data_list in bucket_sample_dict.items():
            ar_name, num_frame = bucket_id[:2]
            if ar_name not in self.profile_results or num_frame not in self.profile_results[ar_name]:
                continue

            if self.generator is not None:
                data_indices = torch.randperm(len(data_list), generator=self.generator).tolist()
                data_list = [data_list[i] for i in data_indices]
                bucket_sample_dict[bucket_id] = data_list

            cur_len = len(data_list)
            sp_size = self.profile_results[ar_name][num_frame]["sp_size"]
            max_bs = self.profile_results[ar_name][num_frame]["max"]["bs"]
            remainder = cur_len % max_bs
            if remainder > 0:
                if self.drop_last:
                    data_list = data_list[:-remainder]
                else:
                    pad = max_bs - remainder
                    if pad > cur_len:
                        data_list = data_list * ((pad + cur_len - 1) // cur_len + 1)
                        data_list = data_list[: pad + cur_len]
                    else:
                        data_list += data_list[:pad]
                bucket_sample_dict[bucket_id] = data_list
                logger.info(f"bucket {bucket_id} has been padded from {cur_len} to {len(data_list)}")
                cur_len = len(data_list)
            num_micro_batches = (cur_len + max_bs - 1) // max_bs

            sp_bucket_id_access_order[sp_size].extend(num_micro_batches * [bucket_id])
            self.effective_samples += cur_len

        return sp_bucket_id_access_order

    def _build_global_bucket_id_access_order(self, sp_bucket_id_access_order):
        wsize = dist.get_world_size()

        # group micro batches to saturate the wsize
        # greedy strategy
        bucket_id_access_order, bucket_list_for_remainder, wsize_for_remainder = [], [], 0
        sp_size_list = sorted(sp_bucket_id_access_order.keys(), reverse=True)
        for sp_size in sp_size_list:
            assert wsize % sp_size == 0, f"sp size {sp_size} cannot divide wsize {wsize}"

            order = sp_bucket_id_access_order[sp_size]
            sort_by_exec_time = sorted(
                order, key=lambda x: self.profile_results[x[0]][x[1]]["max"]["execution_time"], reverse=True
            )

            curr_len = len(sort_by_exec_time)
            consumed_concurrent_micro_batches = 0
            if wsize_for_remainder > 0:
                assert bucket_list_for_remainder, "bucket_list_for_remainder is empty"
                # resolve remainder from previous sp size
                wsize_exclude_remainder = wsize - wsize_for_remainder
                assert wsize_exclude_remainder % sp_size == 0, f"remainder {wsize_exclude_remainder} in sp {sp_size}"

                consumed_concurrent_micro_batches += wsize_exclude_remainder // sp_size
                if consumed_concurrent_micro_batches > curr_len:
                    bucket_list_for_remainder.extend(sort_by_exec_time)
                    wsize_for_remainder += len(sort_by_exec_time) * sp_size
                    continue

                bucket_list_for_remainder.extend(sort_by_exec_time[:consumed_concurrent_micro_batches])
                bucket_id_access_order.append(bucket_list_for_remainder)
                bucket_list_for_remainder, wsize_for_remainder = [], 0

            curr_len = curr_len - consumed_concurrent_micro_batches
            concurrent_micro_batches = wsize // sp_size
            while curr_len >= concurrent_micro_batches:
                bucket_id_access_order.append(
                    sort_by_exec_time[
                        consumed_concurrent_micro_batches : consumed_concurrent_micro_batches + concurrent_micro_batches
                    ]
                )
                consumed_concurrent_micro_batches += concurrent_micro_batches
                curr_len -= concurrent_micro_batches

            if curr_len > 0:
                bucket_list_for_remainder.extend(sort_by_exec_time[consumed_concurrent_micro_batches:])
                wsize_for_remainder += curr_len * sp_size

        return bucket_id_access_order, bucket_list_for_remainder, wsize_for_remainder

    def _build_local_bucket_id_access_order_acc(self, bucket_sample_dict):
        wsize = dist.get_world_size()
        bucket_id_access_order = []
        self.effective_samples = 0

        bucket_sp_map, sp_bucket_map = dict(), dict()
        for bucket_id, data_list in bucket_sample_dict.items():
            ar_name, num_frame = bucket_id[:2]
            if not self.profiler.is_valid_bucket(ar_name, num_frame):
                logger.info(f"skip building batches for bucket {bucket_id} because it's invalid")
                continue

            # collect bucket_sp_map, sp_bucket_map
            sp_size = self.profiler.get_sp_size(ar_name, num_frame)
            max_bs = self.profiler.get_batch_size(ar_name, num_frame)
            cur_len = len(data_list)
            remainder = cur_len % max_bs
            if (not self.keep_last) and remainder > 0:
                if self.drop_last:
                    data_list = data_list[:-remainder]
                else:
                    pad = max_bs - remainder
                    if pad > cur_len:
                        data_list = data_list * ((pad + cur_len - 1) // cur_len + 1)
                        data_list = data_list[: pad + cur_len]
                    else:
                        data_list += data_list[:pad]
            logger.info(f"bucket {bucket_id} original len: {cur_len} padded len: {len(data_list)} for bs {max_bs}")

            bucket_sp_map[bucket_id] = sp_size
            if sp_size not in sp_bucket_map:
                sp_bucket_map[sp_size] = []
            sp_bucket_map[sp_size].append(bucket_id)

            if self.generator is not None:
                data_indices = torch.randperm(len(data_list), generator=self.generator).tolist()
                data_list = [data_list[i] for i in data_indices]

            bucket_sample_dict[bucket_id] = data_list

        bucket_sample_dict_last_access = {k: 0 for k in bucket_sample_dict.keys()}
        sp_size_list = sorted(sp_bucket_map.keys())
        while sp_size_list:
            cur_first_batch_bucket_id_list = []
            remain_gpus = wsize
            has_one_more_batch = True
            while remain_gpus > 0:
                max_sp_idx = 0
                while max_sp_idx < len(sp_size_list) and remain_gpus >= sp_size_list[max_sp_idx]:
                    max_sp_idx += 1

                if max_sp_idx == 0:
                    # if false, cur_first_batch_bucket_id_list will be discarded
                    has_one_more_batch = False

                    # if self.rank == 0:
                    #     log_str = f"exit batch scheduling, cur_first_batch_bucket_id_list: {pformat(cur_first_batch_bucket_id_list, sort_dicts=False)}\n"
                    #     log_str += f"remain_gpus: {remain_gpus}, sp_size_list: {pformat(sp_size_list, sort_dicts=False)}\n"
                    #     log_str += f"sp_bucket_map: {pformat(sp_bucket_map, sort_dicts=False)}\nbucket access:\n"
                    #     for bucket_id in bucket_sample_dict_last_access:
                    #         log_str += f"{bucket_id}: {bucket_sample_dict_last_access[bucket_id]}/{len(bucket_sample_dict[bucket_id])}\n"
                    #     print(f"{log_str}")
                    break

                # select sp size
                if self.generator is not None:
                    cur_sp_size_list = sp_size_list[:max_sp_idx]
                    sp_size_sample_num = OrderedDict({k: len(sp_bucket_map[k]) for k in cur_sp_size_list})
                    total_samples = sum(sp_size_sample_num.values())
                    val = torch.rand(size=(1,), generator=self.generator).item()
                    idx = list(sp_size_sample_num.keys())[-1]
                    for k, v in sp_size_sample_num.items():
                        if val < v / total_samples:
                            idx = k
                            break
                    idx = sp_size_list.index(idx)
                    # idx = torch.randint(low=0, high=max_sp_idx, size=(1,), generator=self.generator).item()
                else:
                    idx = max_sp_idx - 1
                sp = sp_size_list[idx]
                remain_gpus -= sp

                # select bucket id
                if self.generator is not None:
                    bucket_index = torch.randint(
                        low=0, high=len(sp_bucket_map[sp]), size=(1,), generator=self.generator
                    ).item()
                else:
                    bucket_index = 0
                bucket_id = sp_bucket_map[sp][bucket_index]
                ar_name, num_frame = bucket_id[:2]
                # max bs for first batch
                bs = self.profiler.get_batch_size(ar_name, num_frame)

                offset = bucket_sample_dict_last_access[bucket_id]
                num_samples = min(bs, len(bucket_sample_dict[bucket_id]) - offset)
                cur_first_batch_bucket_id_list.append((bucket_id, num_samples))

                offset += num_samples
                bucket_sample_dict_last_access[bucket_id] = offset
                if offset == len(bucket_sample_dict[bucket_id]):
                    sp_bucket_map[sp].pop(bucket_index)
                    if not sp_bucket_map[sp]:
                        sp_size_list.remove(sp)
                        sp_bucket_map.pop(sp)

            if has_one_more_batch:
                # sort to make sure fitting
                cur_first_batch_bucket_id_list = sorted(
                    cur_first_batch_bucket_id_list, key=lambda x: bucket_sp_map[x[0]], reverse=True
                )

                if self.auto_grad_accumulation:
                    exec_time_list = []
                    for bucket_id, bs in cur_first_batch_bucket_id_list:
                        ar_name, num_frame = bucket_id[:2]
                        exec_time_list.append(self.profiler.get_execution_time(ar_name, num_frame))
                    max_time = max(exec_time_list)

                    min_diff = float("inf")
                    num_gas = None
                    for mult in range(1, self.max_grad_accumulation_steps + 1):
                        lcm = max_time * mult

                        cur_gas, cur_diff = [], 0
                        for exec_time in exec_time_list:
                            gas_val = np.round(lcm / exec_time).astype(int).item()
                            cur_diff += abs(gas_val * exec_time - lcm)
                            cur_gas.append(gas_val)
                        if cur_diff < min_diff:
                            min_diff = cur_diff
                            num_gas = cur_gas
                else:
                    num_gas = [1 for _ in cur_first_batch_bucket_id_list]

                # decide accumulate batch_size
                # [[bucket_id, bs] * <num_acc for this seq>] * <num sp group for this iter>
                cur_batch_bucket_id_list = []
                batch_log = []
                # TODO: potential optimization to decide batch size and number of micro batches (for grad acc)
                for bidx, each in enumerate(cur_first_batch_bucket_id_list):
                    this_bucket_acc_list = [each]
                    bucket_id, max_bs = each
                    batch_log.append(
                        [
                            (bucket_id + (bucket_sp_map[bucket_id], max_bs)),
                        ]
                    )
                    # collect effective samples for the first batch of this iter
                    self.effective_samples += max_bs

                    # if has remaining samples for grad acc batch
                    offset = bucket_sample_dict_last_access[bucket_id]
                    total_len = len(bucket_sample_dict[bucket_id])
                    if offset < total_len:
                        ar_name, num_frame = bucket_id[:2]
                        sp = bucket_sp_map[bucket_id]

                        # here I use the max bs from profile
                        bs = max_bs

                        # minus one because of the first batch
                        num_acc = num_gas[bidx] - 1

                        while num_acc > 0 and offset < total_len:
                            num_samples = min(bs, total_len - offset)
                            this_bucket_acc_list.append((bucket_id, num_samples))

                            offset += num_samples
                            num_acc -= 1

                            # collect effective samples for grad acc batches of this iter
                            self.effective_samples += num_samples
                            batch_log[-1].append((bucket_id + (bucket_sp_map[bucket_id], num_samples)))

                        bucket_sample_dict_last_access[bucket_id] = offset
                        # remove exhausted buckets from local indices
                        if offset == total_len:
                            sp_bucket_map[sp].remove(bucket_id)
                            if not sp_bucket_map[sp]:
                                sp_size_list.remove(sp)
                                sp_bucket_map.pop(sp)

                    cur_batch_bucket_id_list.append(this_bucket_acc_list)
                logger.info(
                    f"iter {len(bucket_id_access_order)}, gas: {num_gas} actual: {[len(each) for each in cur_batch_bucket_id_list]}"
                    f", buckets: {batch_log}"
                )
                bucket_id_access_order.append(cur_batch_bucket_id_list)

        return bucket_id_access_order

    def _optimized_schedule_iter(self, bucket_sample_dict, is_global):
        rank, wsize = dist.get_rank(), dist.get_world_size()

        if is_global:
            sp_bucket_id_access_order = self._group_bucket_by_sp_size(bucket_sample_dict)
            # bucket_id_access_order: [bucket_id] * <num sp groups of this iter>
            bucket_id_access_order, _, _ = self._build_global_bucket_id_access_order(sp_bucket_id_access_order)
        else:
            if self.cached_bucket_id_access_order is not None:
                bucket_id_access_order = self.cached_bucket_id_access_order
                self.cached_bucket_id_access_order = None
            else:
                # support grad acc
                # bucket_id_access_order: [[(bucket_id, bs)] * <num acc of this bucket>] * <num sp groups of this iter>
                bucket_id_access_order = self._build_local_bucket_id_access_order_acc(bucket_sample_dict)

        # special care for the remainder, instead of using the minimum sp size, try to use
        # if bucket_list_for_remainder:
        #     pass

        # skip shuffle for bucket id access order. Do we need it?

        num_iter = len(bucket_id_access_order)
        # skip resume code
        start_iter_idx = self.last_micro_batch_access_index

        # generate execution plan
        bucket_last_consumed = defaultdict(int)
        for i in range(start_iter_idx, num_iter):
            bucket_id_access_list = bucket_id_access_order[i]
            sp_size_map_list, bucket_id_map_list = [], []
            for item in bucket_id_access_list:
                if is_global:
                    bucket_id = item
                else:
                    bucket_id = item[0][0]

                sp_size = self.profiler.get_sp_size(bucket_id[0], bucket_id[1])

                sp_size_map_list.extend([sp_size] * sp_size)
                bucket_id_map_list.extend([item] * sp_size)

            # essentially, we want to set the sp size for the current iteration before running training,
            # however, when using multiple workers and prefetch in dataloader,
            # simply set_sp_size(sp_size) can cause inconsistency in the sp size
            assert len(sp_size_map_list) == wsize
            # append_sp_size(sp_size_map_list[rank])
            sp_size = sp_size_map_list[rank]

            if is_global:
                gas = 1
                bucket_id = bucket_id_map_list[rank]
                max_bs = self.profiler.get_batch_size(bucket_id[0], bucket_id[1])
                cur_idx = bucket_last_consumed[bucket_id]
                cur_offset = min(max_bs, len(bucket_sample_dict[bucket_id]) - cur_idx)
                cur_micro_batches = bucket_sample_dict[bucket_id][cur_idx : cur_idx + cur_offset]
                bucket_last_consumed[bucket_id] += cur_offset

                real_t, real_h, real_w = self.bucket.get_thw(bucket_id)
                cur_micro_batches = [
                    (idx, real_t, real_h, real_w, bucket_id[0], sp_size, gas) for idx in cur_micro_batches
                ]
                yield cur_micro_batches
            else:
                # append_num_grad_acc(len(bucket_id_map_list[rank]))
                bucket_list = bucket_id_map_list[rank]
                gas = len(bucket_list)
                cur_micro_batches = []
                for each in bucket_list:
                    bucket_id, bs = each
                    cur_idx = bucket_last_consumed[bucket_id]
                    cur_offset = min(bs, len(bucket_sample_dict[bucket_id]) - cur_idx)
                    gas_micro_batches = bucket_sample_dict[bucket_id][cur_idx : cur_idx + cur_offset]
                    bucket_last_consumed[bucket_id] += cur_offset

                    real_t, real_h, real_w = self.bucket.get_thw(bucket_id)
                    cur_micro_batches.extend(
                        [(idx, real_t, real_h, real_w, bucket_id[0], sp_size, gas) for idx in gas_micro_batches]
                    )
                yield cur_micro_batches

        self.reset()

    def get_num_batch_with_optimized_schedule(self, bucket_sample_dict, is_global) -> int:
        if is_global:
            sp_bucket_id_access_order = self._group_bucket_by_sp_size(bucket_sample_dict)
            bucket_id_access_order, _, _ = self._build_global_bucket_id_access_order(sp_bucket_id_access_order)
        else:
            bucket_id_access_order = self._build_local_bucket_id_access_order_acc(bucket_sample_dict)
            self.cached_bucket_id_access_order = bucket_id_access_order
        self.approximate_num_batch = len(bucket_id_access_order)

        # collect statistics
        total_samples = 0
        bucket_stat_dict = dict()
        for k, v in bucket_sample_dict.items():
            ar_name, num_frame = k[:2]
            if not self.profiler.is_valid_bucket(ar_name, num_frame):
                continue
            size = len(v)
            max_bs = self.profiler.get_batch_size(ar_name, num_frame)
            if self.keep_last or (not self.drop_last):
                effect_size = size + max_bs - 1
            else:
                effect_size = size
            num_batch = effect_size // max_bs
            if not self.keep_last:
                size = max_bs * num_batch

            total_samples += size

            bucket_stat_dict[k] = [size, num_batch]

        # log
        if dist.get_rank() == 0 and self.verbose:
            logger.info(f"Bucket Info at epoch {self.epoch} with optimized schedule:")
            logger.info("Bucket [#sample] by HxWxT:\n%s", pformat(bucket_stat_dict, sort_dicts=False))
            logger.info(
                "#training batch: %s, #training sample: %s, #non empty bucket: %s",
                self.approximate_num_batch,
                total_samples,
                len(bucket_sample_dict),
            )

        return self.approximate_num_batch

    def get_num_batch(self, bucket_sample_dict) -> int:
        bucket_id_access_order = self._build_bucketized_bucket_id_access_order(bucket_sample_dict)
        self.cached_bucket_id_access_order = bucket_id_access_order
        self.approximate_num_batch = len(bucket_id_access_order)

        # collect statistics
        total_samples = 0
        total_batch = 0

        bucket_stat_dict = dict()
        for k, v in bucket_sample_dict.items():
            if not self.profiler.is_valid_bucket(k[0], k[1]):
                continue
            size = len(v)
            bs = self.get_batch_size(k)
            if self.keep_last or (not self.drop_last):
                effect_size = size + bs - 1
            else:
                effect_size = size
            num_batch = effect_size // bs
            if not self.keep_last:
                size = bs * num_batch

            total_samples += size
            total_batch += num_batch

            bucket_stat_dict[k] = [size, num_batch]

        # log
        if dist.get_rank() == 0 and self.verbose:
            logger.info(f"Bucket Info at epoch {self.epoch} with bucketized schedule:")
            logger.info("Bucket [#sample] by HxWxT:\n%s", pformat(bucket_stat_dict, sort_dicts=False))
            logger.info(
                "#training batch: %s, #training sample: %s, #non empty bucket: %s",
                total_batch,
                total_samples,
                len(bucket_sample_dict),
            )
        return self.approximate_num_batch

    def reset(self):
        self.last_micro_batch_access_index = 0

    def state_dict(self, num_steps: int) -> dict:
        # the last_micro_batch_access_index in the __iter__ is often
        # not accurate during multi-workers and data prefetching
        # thus, we need the user to pass the actual steps which have been executed
        # to calculate the correct last_micro_batch_access_index
        return {"seed": self.seed, "epoch": self.epoch, "last_micro_batch_access_index": num_steps * self.num_replicas}

    def load_state_dict(self, state_dict: dict) -> None:
        self.__dict__.update(state_dict)

    def profile(
        self,
        vae,
        mask_generator,
        model,
        scheduler,
        optimizer,
        lr_scheduler,
        text_encoder_model_max_length,
        text_encoder_output_dim,
        device,
        dtype,
        dump_dir,
        alloc_memory_fraction,
        profile_path=None,
        end2end_profile=False,
        dynamic_recompute=False,
        grad_acc=False,
    ):
        if profile_path is not None and os.path.exists(profile_path):
            with open(profile_path) as f:
                self.profile_results = json.load(f)
            # num_frame will be read as str, and we need to convert it back to int
            for ar_name in self.profile_results:
                self.profile_results[ar_name] = {int(k): v for k, v in self.profile_results[ar_name].items()}

        else:
            if end2end_profile:
                profile_func = self._end2end_profile
            else:
                profile_func = partial(self._fast_profile, dynamic_recompute=dynamic_recompute, grad_acc=grad_acc)

            # with PrintingMode():
            profile_func(
                vae,
                mask_generator,
                model,
                scheduler,
                optimizer,
                lr_scheduler,
                text_encoder_model_max_length,
                text_encoder_output_dim,
                device,
                dtype,
                dump_dir,
                alloc_memory_fraction,
            )

        if dynamic_recompute:
            for ar_name in self.profile_results:
                for num_frame in self.profile_results[ar_name]:
                    assert (
                        "recompute_cfg" in self.profile_results[ar_name][num_frame]
                    ), f"recompute_cfg not found for {ar_name} {num_frame}"
                    recompute_dict = self.profile_results[ar_name][num_frame]["recompute_cfg"]
                    recompute_cfg = []
                    for d in range(model.module.config.depth):
                        for prefix in ["spatial_", "temporal_"]:
                            if d < recompute_dict[prefix + "self_attn"]:  # not recompute spatial self attn
                                if d < recompute_dict[prefix + "cross_attn"]:  # not recompute spatial cross attn
                                    if d < recompute_dict[prefix + "mlp"]:  # not recompute spatial mlp
                                        recompute_cfg.append(STDiT3BlockRecomputeConfig.NONE)
                                    else:  # recompute spatial mlp
                                        recompute_cfg.append(STDiT3BlockRecomputeConfig.MLP)
                                else:  # recompute spatial cross attn
                                    if d < recompute_dict[prefix + "mlp"]:  # not recompute spatial mlp
                                        recompute_cfg.append(STDiT3BlockRecomputeConfig.CROSS_ATTN)
                                    else:  # recompute spatial mlp
                                        recompute_cfg.append(STDiT3BlockRecomputeConfig.CROSS_ATTN_AND_MLP)
                            else:  # recompute spatial self attn
                                if d < recompute_dict[prefix + "cross_attn"]:  # not recompute spatial cross attn
                                    if d < recompute_dict[prefix + "mlp"]:  # not recompute spatial mlp
                                        recompute_cfg.append(STDiT3BlockRecomputeConfig.SELF_ATTN)
                                    else:  # recompute spatial mlp
                                        recompute_cfg.append(STDiT3BlockRecomputeConfig.SELF_ATTN_AND_MLP)
                                else:  # recompute spatial cross attn
                                    if d < recompute_dict[prefix + "mlp"]:
                                        recompute_cfg.append(STDiT3BlockRecomputeConfig.SELF_AND_CROSS_ATTN)
                                    else:  # recompute spatial mlp
                                        recompute_cfg.append(STDiT3BlockRecomputeConfig.BLOCK)

                    self.profile_results[ar_name][num_frame]["recompute_cfg2"] = recompute_cfg
        logger.info(f"Profile results: {pformat(self.profile_results)}")

    def _end2end_profile(
        self,
        vae,
        mask_generator,
        model,
        scheduler,
        optimizer,
        lr_scheduler,
        text_encoder_model_max_length,
        text_encoder_output_dim,
        device,
        dtype,
        dump_dir,
        alloc_memory_fraction,
    ):
        """
        `available_memory_fraction`: an empirical value (0, 1) to avoid comm deadlock for dynamic sp.
            If you meet deadlock during running this profiling, try to reduce this value.
        """
        torch.cuda.set_per_process_memory_fraction(alloc_memory_fraction)
        rank, wsize = dist.get_rank(), dist.get_world_size()
        max_sp = torch.cuda.device_count()

        # warmup with the smallest input data and all gpu devices
        height, width = DEFAULT_AR_MAP["144p"]
        num_frame, bs = 51, 1
        self.parallel_mgr.set_sp_size(wsize)

        for _ in range(1):
            x = torch.rand(bs, 3, num_frame, height, width, device=device, dtype=dtype)
            with torch.no_grad():
                x = vae.encode(x)
            model_args = {
                "y": torch.rand(
                    (bs, 1, text_encoder_model_max_length, text_encoder_output_dim), device=device, dtype=dtype
                ),
                "mask": torch.ones((bs, text_encoder_model_max_length), device=device, dtype=torch.long),
            }
            model_args["height"] = torch.tensor([height] * bs, device=device, dtype=dtype)
            model_args["width"] = torch.tensor([width] * bs, device=device, dtype=dtype)
            model_args["num_frames"] = torch.tensor([num_frame] * bs, device=device, dtype=dtype)
            model_args["fps"] = torch.tensor([30 if num_frame > 1 else 120] * bs, device=device, dtype=dtype)
            model_args["ar"] = torch.tensor([height / width] * bs, device=device, dtype=dtype)

            mask = None
            if mask_generator is not None:
                mask = mask_generator.get_masks(x)
                model_args["x_mask"] = mask
            loss_dict = scheduler.training_losses(model, x, model_args, mask=mask)
            loss = loss_dict["loss"].mean()
            model.backward(loss)
            model.step()
            # optimizer.zero_grad()
            # update learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()

        print_memory_stats("Warmup")

        profile_results = {}
        detail_results = []
        timers = {}
        timer_keys = [
            # "move_data",
            # "encode",
            # "mask",
            "diffusion",
            "backward",
            # "update",
            "iteration",
        ]
        for key in timer_keys:
            timers[key] = GroupTimer(key)  # , log=True)

        def profile_iter():
            timers["iteration"].__enter__()

            row = [ar_name, num_frame, bs, cur_size]
            _, _, initial_memory, initial_cache, _, _ = print_memory_stats(
                f"reset sp to {cur_size} for bucket {ar_name} {num_frame} {bs}"
            )

            # with timers["move_data"] as move_data_t:
            nf = 1
            if num_frame > 1:
                nf = num_frame * 5 // 17
            x = torch.rand(bs, 4, nf, height // 8, width // 8, device=device, dtype=dtype)
            # run with the longest caption
            model_args = {
                "y": torch.rand(
                    (bs, 1, text_encoder_model_max_length, text_encoder_output_dim),
                    device=device,
                    dtype=dtype,
                ),
                "mask": torch.ones((bs, text_encoder_model_max_length), device=device, dtype=torch.long),
            }

            model_args["height"] = torch.tensor([height] * bs, device=device, dtype=dtype)
            model_args["width"] = torch.tensor([width] * bs, device=device, dtype=dtype)
            model_args["num_frames"] = torch.tensor([num_frame] * bs, device=device, dtype=dtype)
            model_args["fps"] = torch.tensor([30 if num_frame > 1 else 120] * bs, device=device, dtype=dtype)
            model_args["ar"] = torch.tensor([height / width] * bs, device=device, dtype=dtype)
            # row.append(move_data_t.elapsed_time)

            # with timers["encode"] as encode_t:
            #     with torch.no_grad():
            #         x = vae.encode(x)
            # row.append(encode_t.elapsed_time)

            # with timers["mask"] as mask_t:
            mask = None
            if mask_generator is not None:
                mask = mask_generator.get_masks(x)
                model_args["x_mask"] = mask
            # row.append(mask_t.elapsed_time)

            with timers["diffusion"] as diffusion_t:
                loss_dict = scheduler.training_losses(model, x, model_args, mask=mask)
            row.append(diffusion_t.elapsed_time)

            with timers["backward"] as reduce_loss_t:
                loss = loss_dict["loss"].mean()
                model.backward(loss)
                # booster.backward(loss=loss, optimizer=optimizer)
            row.append(reduce_loss_t.elapsed_time)

            # with timers["update"] as update_t:
            # optimizer.step()
            # optimizer.zero_grad()
            model.step()
            # update learning rate
            if lr_scheduler is not None:
                lr_scheduler.step()
            # row.append(update_t.elapsed_time)

            timers["iteration"].__exit__(0, 0, 0)
            row.append(timers["iteration"].elapsed_time)
            row.extend(
                [
                    initial_memory / GB,
                    torch.cuda.max_memory_allocated() / GB,
                    initial_cache / GB,
                    torch.cuda.max_memory_reserved() / GB,
                ]
            )
            return row

        dist.barrier()
        profile_timer = GroupTimer("profile", group=self.parallel_mgr.dp_group)
        profile_timer.__enter__()

        result_row = []
        for ar_name, nframe2bsize in self.bucket.bucket_bs.items():
            # use the default ar to approximate the sequence length
            height, width = DEFAULT_AR_MAP[ar_name]
            for num_frame, max_bs in nframe2bsize.items():
                if max_bs is None:
                    continue
                # Given ar_name, num_frame
                # T1: find the minimum sp_size to run with bs=1
                bs, cur_size, is_success = 1, 1, False
                while cur_size <= max_sp and not is_success:
                    try:
                        clean_cache()
                        self.parallel_mgr.set_sp_size(cur_size)
                        self.change_timer_group(timers)
                        result_row = profile_iter()
                        is_success = True
                    except torch.cuda.OutOfMemoryError as e:
                        reset_status(model, optimizer)

                        logger.info(
                            f">>> [Profiling] skip sp {cur_size} for bucket: {ar_name} {num_frame} {bs} due to OOM"
                        )
                        cur_size *= 2
                        continue

                if not is_success:
                    logger.info(f">>> [Profiling] bucket {ar_name} {num_frame} cannot fit into the cluster")
                    continue

                detail_results.append(result_row)

                if ar_name not in profile_results:
                    profile_results[ar_name] = {}

                profile_results[ar_name][num_frame] = {
                    "sp_size": cur_size,
                    # "min": {
                    #     "bs": bs,
                    #     "execution_time": result_row[-5],
                    #     "memory_consumed": result_row[-3],
                    # },
                }

                logger.info(f">>> [Profiling {rank}] bucket: {ar_name} {num_frame} finish T1 at sp {cur_size}")

                # T2: find the maximum bs with the minimum sp_size
                prev_bs = bs
                bs *= 2
                while True:
                    try:
                        clean_cache()
                        result_row = profile_iter()
                        detail_results.append(result_row)
                        prev_bs = bs
                        bs *= 2
                        logger.info(f">>> [Profiling {rank}] Bucket {ar_name} {num_frame} pass bs {prev_bs}")
                    except torch.cuda.OutOfMemoryError as e:
                        reset_status(model, optimizer)

                        logger.info(f">>> [Profiling {rank}] Bucket {ar_name} {num_frame} finish T2 at bs {prev_bs}")
                        break

                profile_results[ar_name][num_frame]["max"] = {
                    "bs": prev_bs,
                    "execution_time": result_row[-5],
                    "memory_consumed": result_row[-3],
                }

        profile_timer.__exit__(0, 0, 0)
        logger.info(
            f">>> [Profiling] Profile results: {pformat(profile_results, sort_dicts=False)}\n"
            f">>> [Profiling] Profile cost: {profile_timer.elapsed_time:.2f} s"
        )
        if rank == 0:
            df = pd.DataFrame(
                detail_results,
                columns=["ar", "num_frame", "bs", "sp_size"]
                + timer_keys
                + ["alloc", "max_alloc", "reserved", "max_reserved"],
            )
            df.to_csv(f"{dump_dir}/detail_profile.csv", index=False)

            with open(f"{dump_dir}/profile.json", "w") as f:
                json.dump(profile_results, f)
            logger.info(df)

        send_list = [profile_results]
        dist.broadcast_object_list(send_list, src=0)
        self.profile_results = send_list[0]

        torch.cuda.set_per_process_memory_fraction(1.0)
        clean_cache()

    def _fast_profile(
        self,
        vae,
        mask_generator,
        model,
        scheduler,
        optimizer,
        lr_scheduler,
        text_encoder_model_max_length,
        text_encoder_output_dim,
        device,
        dtype,
        dump_dir,
        alloc_memory_fraction,
        dynamic_recompute,
        grad_acc,
    ):
        _, total, _, _, _, _ = print_memory_stats("entering profiling")
        # in case of memory fragmentation
        memory_cap = total * alloc_memory_fraction
        torch.cuda.set_per_process_memory_fraction(alloc_memory_fraction)
        rank = dist.get_rank()
        max_sp = torch.cuda.device_count()

        total_depth = model.module.config.depth
        valid_depth = 2
        profile_results, detail_results, raw_results = {}, [], []
        profile_ctx = None

        org_recompute_cfg = []
        for d in range(valid_depth):
            org_recompute_cfg.append(
                (model.module.spatial_blocks[d].recompute_cfg, model.module.temporal_blocks[d].recompute_cfg)
            )
            model.module.spatial_blocks[d].recompute_cfg = STDiT3BlockRecomputeConfig.BLOCK
            model.module.temporal_blocks[d].recompute_cfg = STDiT3BlockRecomputeConfig.BLOCK

        if grad_acc:
            model.set_train_batch_size(model.train_micro_batch_size_per_gpu() * model.dp_world_size * 100000)
            model.optimizer.gradient_accumulation_steps = 100000

        timers = {}
        timer_keys = [
            "diffusion",
            "backward",
            "iteration",
        ]
        for key in timer_keys:
            timers[key] = GroupTimer(key)

        def profile_iter():
            _, _, initial_memory, initial_cache, _, _ = print_memory_stats(
                f"START bucket {ar_name} {num_frame} {bs} with sp {cur_size}"
            )
            row = [ar_name, num_frame, bs, cur_size]

            # move_data
            nf = 1
            if num_frame > 1:
                nf = num_frame * 5 // 17
            x = torch.rand(bs, 4, nf, height // 8, width // 8, device=device, dtype=dtype)
            # run with the longest caption
            model_args = {
                "y": torch.rand(
                    (bs, 1, text_encoder_model_max_length, text_encoder_output_dim),
                    device=device,
                    dtype=dtype,
                ),
                "mask": torch.ones((bs, text_encoder_model_max_length), device=device, dtype=torch.long),
            }

            model_args["height"] = torch.tensor([height] * bs, device=device, dtype=dtype)
            model_args["width"] = torch.tensor([width] * bs, device=device, dtype=dtype)
            model_args["num_frames"] = torch.tensor([num_frame] * bs, device=device, dtype=dtype)
            model_args["fps"] = torch.tensor([30 if num_frame > 1 else 120] * bs, device=device, dtype=dtype)
            model_args["ar"] = torch.tensor([height / width] * bs, device=device, dtype=dtype)
            # execute two layer, one for warmup, one for profiling
            model_args["valid_depth"] = valid_depth

            timers["iteration"].__enter__()

            # mask
            mask = None
            if mask_generator is not None:
                mask = mask_generator.get_masks(x)
                model_args["x_mask"] = mask

            # diffusion
            with timers["diffusion"] as diffusion_t:
                loss_dict = scheduler.training_losses(model, x, model_args, mask=mask)
            row.append(diffusion_t.elapsed_time)

            # backward
            with timers["backward"] as reduce_loss_t:
                loss = loss_dict["loss"].mean()
                model.backward(loss)
            row.append(reduce_loss_t.elapsed_time)

            # update
            model.step()
            if lr_scheduler is not None:
                lr_scheduler.step()

            timers["iteration"].__exit__(0, 0, 0)
            row.append(timers["iteration"].elapsed_time)
            row.extend(
                [initial_memory, torch.cuda.max_memory_allocated(), initial_cache, torch.cuda.max_memory_reserved()]
            )

            if profile_ctx is not None:
                row.extend(profile_ctx.get_profile_results())

            return row

        def estimate_overhead(data_row):
            # spatial & temporal [self_attn, cross_attn, mlp]
            fwd_time = sum(data_row[11:17])
            bwd_time = sum(data_row[17:23])
            layer_time = fwd_time * 2 + bwd_time
            # spatial & temporal input memory
            layer_memory = sum(data_row[23:25])
            missing_gradient_per_layer = 0
            if grad_acc:
                # STDiT block * (spatial + temporal) * 2 bytes
                missing_gradient_per_layer = 21255696 * 2 * 2
            # model with one-layer time/memory + one-layer time/memory * (total_depth - 1)
            pred_full_time = data_row[6] + layer_time * (total_depth - valid_depth)
            pred_full_mem = data_row[8] + (layer_memory + missing_gradient_per_layer) * (total_depth - valid_depth)
            return pred_full_time, pred_full_mem

        # WARMUP with the smallest input data and all gpu devices
        ar_name = "144p"
        height, width = DEFAULT_AR_MAP[ar_name]
        num_frame, bs = 51, 1
        cur_size = max_sp
        self.parallel_mgr.set_sp_size(cur_size)
        profile_iter()
        print_memory_stats("Warmup")

        enable_profile()
        profile_ctx = get_profile_context()
        submodule_fields = profile_ctx.get_submodule_fields()
        num_modules = len(submodule_fields)

        dist.barrier()
        profile_timer = GroupTimer("profile", group=self.parallel_mgr.dp_group)
        profile_timer.__enter__()

        result_row: list
        latest_raw_row = None
        # PROFILE
        for ar_name, nframe2bsize in self.bucket.bucket_bs.items():
            # use the default ar to approximate the sequence length
            height, width = DEFAULT_AR_MAP[ar_name]
            for num_frame, max_bs in nframe2bsize.items():
                if max_bs is None:
                    continue

                # baseline
                if not dynamic_recompute and not self.dynamic_sp:
                    fix_sp = self.parallel_mgr.get_sp_size()
                    bs, cur_size, is_success = 1, fix_sp, False

                    # find the max bs
                    while True:
                        prev_bs = bs
                        bs *= 2
                        pass_depth_loop = True

                        try:
                            clean_cache()
                            profile_iter()
                            raw_result_row = profile_iter()
                        except torch.cuda.OutOfMemoryError as e:
                            reset_status(model, optimizer)
                            pass_depth_loop = False

                        is_success = False
                        if pass_depth_loop:
                            pred_full_time, pred_full_mem = estimate_overhead(raw_result_row)
                            logger.info(
                                f">>> [Profiling] {ar_name} {num_frame} {bs} at sp {cur_size}: {pred_full_mem/GB:.2f}/{memory_cap/GB:.2f} GB, {pred_full_time:.2f} s"
                            )
                            if pred_full_mem <= memory_cap:
                                is_success = True
                                # raw_results.append(raw_result_row)
                                latest_raw_row = raw_result_row
                                logger.info(f">>> [Profiling] DONE bucket {ar_name} {num_frame} {bs} at sp {cur_size}")

                        if not is_success:
                            logger.info(
                                f">>> [Profiling] BEST bs for bucket {ar_name} {num_frame} is {prev_bs} at sp {cur_size}"
                            )
                            break
                    bs = prev_bs  # the best (micro) bs for this bucket
                    raw_results.append(latest_raw_row)

                    # This is for validating if the estimation works.
                    # I verify with 4*A100 40G that time diff < 10%, memory diff < 1% compared to end2end profile
                    pred_full_time, pred_full_mem = estimate_overhead(latest_raw_row)
                    if ar_name not in profile_results:
                        profile_results[ar_name] = {}
                    profile_results[ar_name][num_frame] = dict(
                        sp_size=cur_size,
                        max=dict(bs=bs, execution_time=pred_full_time, memory_consumed=pred_full_mem / GB),
                    )
                else:
                    dp_results = []
                    cur_size = 1
                    while cur_size <= max_sp:  # search through sp dimension
                        self.parallel_mgr.set_sp_size(cur_size)
                        self.change_timer_group(timers)

                        bs = 1
                        is_success = True
                        while is_success:  # search through bs dimension
                            try:
                                clean_cache()
                                profile_iter()
                                raw_result_row = profile_iter()
                            except torch.cuda.OutOfMemoryError as e:
                                reset_status(model, optimizer)
                                is_success = False
                                logger.info(
                                    f">>> [Profiling] Bucket {ar_name} {num_frame} at {bs} sp {cur_size} doesn't pass profile, OOM!"
                                )

                            if is_success:
                                pred_full_time, pred_full_mem = estimate_overhead(raw_result_row)
                                if pred_full_mem <= memory_cap:
                                    raw_results.append(raw_result_row)

                                    avail_mem = int(np.floor((memory_cap - pred_full_mem) / GB))
                                    if dynamic_recompute:
                                        # planning
                                        fwd_time_offset = 11
                                        memory_offset = 25

                                        dp = torch.zeros(avail_mem + 1, dtype=torch.float)
                                        trace = torch.zeros((avail_mem + 1, num_modules), dtype=torch.int)

                                        for i in range(num_modules):
                                            module_time_cost = raw_result_row[fwd_time_offset + i]
                                            module_mem_cost_in_bytes = raw_result_row[memory_offset + i]

                                            temp_dp = dp.clone()
                                            temp_trace = trace.clone()

                                            for cnt in range(1, total_depth + 1):
                                                time_cost = module_time_cost * cnt
                                                mem_cost = int(np.ceil(module_mem_cost_in_bytes * cnt / GB))

                                                for cur_mem in range(mem_cost, avail_mem + 1):
                                                    if temp_dp[cur_mem] < dp[cur_mem - mem_cost] + time_cost:
                                                        temp_dp[cur_mem] = dp[cur_mem - mem_cost] + time_cost
                                                        temp_trace[cur_mem] = trace[cur_mem - mem_cost].clone()
                                                        temp_trace[cur_mem, i] = cnt

                                            dp = temp_dp
                                            trace = temp_trace

                                        reduced_time = dp[avail_mem].item()
                                        best_full_time = pred_full_time - reduced_time
                                        best_trace = trace[avail_mem].tolist()
                                        for i in range(num_modules):
                                            avail_mem -= int(
                                                np.ceil((raw_result_row[memory_offset + i] * best_trace[i]) / GB)
                                            )

                                        assert avail_mem >= 0, f"rest memory: {avail_mem}, trace: {best_trace}"
                                        result_row = (
                                            raw_result_row[:4]
                                            + [best_full_time, memory_cap / GB - avail_mem, reduced_time, avail_mem]
                                            + best_trace
                                        )
                                    else:  # dynamic sp only
                                        result_row = raw_result_row[:4] + [
                                            pred_full_time,
                                            pred_full_mem / GB,
                                            0,
                                            avail_mem,
                                        ]

                                    dp_results.append(result_row)
                                    detail_results.append(result_row)

                                    logger.info(
                                        f">>> [Profiling] DONE BS search for bucket {ar_name} {num_frame} at {bs} sp {cur_size}"
                                    )
                                else:
                                    is_success = False
                                    logger.info(
                                        f">>> [Profiling] Bucket {ar_name} {num_frame} at {bs} sp {cur_size} pass profile but exceed memory limit: {pred_full_mem/GB:.2f}/{memory_cap/GB:.2f} GB"
                                    )

                            if not is_success:
                                logger.info(
                                    f">>> [Profiling] STOP BS search for bucket {ar_name} {num_frame} at {bs} sp {cur_size}"
                                )
                            bs *= 2

                        cur_size *= 2

                    if not dp_results:
                        logger.info(
                            f">>> [Profiling] SKIP bucket {ar_name} {num_frame} which cannot fit into the cluster"
                        )
                        continue

                    # criterion: max throughput = (bs / sp_size / iter_time)
                    best = sorted(dp_results, key=lambda x: x[2] / x[3] / x[4], reverse=True)[0]

                    if ar_name not in profile_results:
                        profile_results[ar_name] = {}
                    profile_results[ar_name][num_frame] = dict(
                        sp_size=best[3],
                        max=dict(bs=best[2], execution_time=best[4], memory_consumed=best[5]),
                    )
                    if dynamic_recompute:
                        profile_results[ar_name][num_frame]["recompute_cfg"] = {
                            k: v for k, v in zip(submodule_fields, best[8:])
                        }

        profile_timer.__exit__(0, 0, 0)

        if not dynamic_recompute and not grad_acc:
            # balance the execution time of each bucket by adjusting batch size
            logger.info(
                f">>> [Profiling] Profile results before adjustment: {pformat(profile_results, sort_dicts=False)}\n"
                f">>> [Profiling] Profile cost before adjustment: {profile_timer.elapsed_time:.2f} s"
            )

            def score_func(new_time, median_time):
                if not self.dynamic_sp:
                    return torch.abs(new_time - median_time)
                if new_time > median_time:
                    return (new_time - median_time) * 1.5
                else:
                    return (median_time - new_time) * 1

            time_list = []
            for ar_name, num_frame_dict in profile_results.items():
                for num_frame, info in num_frame_dict.items():
                    time_list.append(info["max"]["execution_time"])
            median_time = np.median(time_list)

            for ar_name, num_frame_dict in profile_results.items():
                for num_frame, info in num_frame_dict.items():
                    cur_time = info["max"]["execution_time"]
                    cur_bs = info["max"]["bs"]
                    cur_diff = score_func(cur_time, median_time)
                    if cur_time > median_time:
                        if self.dynamic_sp and cur_bs == 1:
                            while cur_time > median_time:
                                tmp_sp_size = info["sp_size"] * 2
                                cur_time = cur_time / 2
                                if tmp_sp_size > max_sp:
                                    break
                            new_diff = score_func(cur_time, median_time)
                            if new_diff < cur_diff:
                                info["sp_size"] = tmp_sp_size
                                info["max"]["execution_time"] = cur_time
                        else:
                            new_bs = max(1, int(cur_bs * median_time / cur_time))
                            new_time = cur_time * new_bs / cur_bs
                            new_diff = score_func(new_time, median_time)
                            if new_diff < cur_diff or not self.dynamic_sp:
                                info["max"]["execution_time"] = cur_time * new_bs / cur_bs
                                info["max"]["bs"] = new_bs

        logger.info(
            f">>> [Profiling] Profile results: {pformat(profile_results, sort_dicts=False)}\n"
            f">>> [Profiling] Profile cost: {profile_timer.elapsed_time:.2f} s"
        )
        if rank == 0:
            df = pd.DataFrame(
                raw_results,
                columns=["ar", "num_frame", "bs", "sp_size"]
                + timer_keys
                + ["alloc", "max_alloc", "reserved", "max_reserved"]
                + profile_ctx.get_profile_fields(),
            )
            df.to_csv(f"{dump_dir}/raw_results.csv", index=False)

            detail_df = pd.DataFrame(
                detail_results,
                columns=["ar", "num_frame", "bs", "sp_size"]
                + ["pred_time", "pred_memory", "saved_time", "cost_memory"]
                + ([f"{k}_cnt" for k in submodule_fields] if dynamic_recompute else []),
            )
            detail_df.to_csv(f"{dump_dir}/detail_profile.csv", index=False)
            logger.info(detail_df)

            with open(f"{dump_dir}/profile.json", "w") as f:
                json.dump(profile_results, f)

        send_list = [profile_results]
        dist.broadcast_object_list(send_list, src=0)
        self.profile_results = send_list[0]

        if grad_acc:
            model.set_train_batch_size(model.train_micro_batch_size_per_gpu() * model.dp_world_size * 1)
            model.optimizer.gradient_accumulation_steps = 1
            model.zero_grad()

        for d in range(valid_depth):
            org_recompute_cfg.append(
                (model.module.spatial_blocks[d].recompute_cfg, model.module.temporal_blocks[d].recompute_cfg)
            )
            model.module.spatial_blocks[d].recompute_cfg = org_recompute_cfg[d][0]
            model.module.temporal_blocks[d].recompute_cfg = org_recompute_cfg[d][1]

        torch.cuda.set_per_process_memory_fraction(1.0)
        clean_cache()

        disable_profile()


class BatchDistributedSampler(DistributedSampler):
    """
    Used with BatchDataset;
    Suppose len_buffer == 5, num_buffers == 6, #GPUs == 3, then
           | buffer {i}          | buffer {i+1}
    ------ | ------------------- | -------------------
    rank 0 |  0,  1,  2,  3,  4, |  5,  6,  7,  8,  9
    rank 1 | 10, 11, 12, 13, 14, | 15, 16, 17, 18, 19
    rank 2 | 20, 21, 22, 23, 24, | 25, 26, 27, 28, 29
    """

    def __init__(self, dataset: Dataset, **kwargs):
        super().__init__(dataset, **kwargs)
        self.start_index = 0

    def __iter__(self):
        num_buffers = self.dataset.num_buffers
        len_buffer = self.dataset.len_buffer
        num_buffers_i = num_buffers // self.num_replicas
        num_samples_i = len_buffer * num_buffers_i

        indices_i = np.arange(self.start_index, num_samples_i) + self.rank * num_samples_i
        indices_i = indices_i.tolist()

        return iter(indices_i)

    def reset(self):
        self.start_index = 0

    def state_dict(self, step) -> dict:
        return {"start_index": step}

    def load_state_dict(self, state_dict: dict):
        self.start_index = state_dict["start_index"] + 1
