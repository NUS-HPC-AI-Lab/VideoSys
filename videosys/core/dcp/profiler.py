import json
import logging
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from itertools import accumulate
from pprint import pformat
from typing import Dict, List, NamedTuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm

from videosys.core.dcp.recompute import disable_profile, enable_profile, get_profile_context
from videosys.core.distributed.parallel_mgr import DynamicParallelManager
from videosys.training.datasets.open_sora.aspect import ASPECT_RATIOS, DEFAULT_AR_MAP
from videosys.utils.training import GroupTimer, set_grad_accumulation_steps

PROFILER = None
GB = 1024**3


def clean_cache():
    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def reset_status(model):
    optimizer = model.optimizer
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


class DataPlan(NamedTuple):
    ar_name: str
    num_frame: int
    sp_size: int
    gas: int
    bs: int
    warmup_iter: bool


@dataclass
class ProfileResult:
    ar_name: str
    num_frame: int
    bs: int
    sp_size: int
    execution_time: float
    max_alloc_memory: int

    input_memory: List[int] = None
    submodule_fields: List[str] = None
    submodule_fwd_time: List[float] = None
    submodule_bwd_time: List[float] = None
    submodule_memory: List[int] = None

    def to_list(self):
        ret = (
            [
                self.ar_name,
                self.num_frame,
                self.bs,
                self.sp_size,
                self.execution_time,
                self.max_alloc_memory,
            ]
            + self.submodule_fwd_time
            + self.submodule_bwd_time
            + self.input_memory
            + self.submodule_memory
        )
        return ret


class ProfileDataIter:
    def __init__(self, profiler):
        self.profiler: Profiler = profiler

        self.data_plan = [
            DataPlan(
                ar_name="144p",
                num_frame=51,
                sp_size=self.profiler.max_sp,
                gas=1,
                bs=1,
                warmup_iter=False,
            )
        ]
        self.next_idx = 0

    def __iter__(self):
        while self.next_idx < len(self.data_plan):
            data_plan = self.data_plan[self.next_idx]
            data_idx = self.next_idx
            self.next_idx += 1

            height, width = DEFAULT_AR_MAP[data_plan.ar_name]
            nf = 1
            if data_plan.num_frame > 1:
                nf = data_plan.num_frame * 5 // 17

            ret = dict(
                ar_name=data_plan.ar_name,
                num_frame=data_plan.num_frame,
                sp_size=data_plan.sp_size,
                gas=data_plan.gas,
                data=[],
                profile_grad_acc=self.profiler.auto_grad_acc and data_idx > 0,
            )

            for _ in range(data_plan.gas):
                ret["data"].append(
                    dict(
                        video=torch.rand(data_plan.bs, 4, nf, height // 8, width // 8),
                        text=torch.rand(
                            data_plan.bs,
                            1,
                            self.profiler.text_max_seq_len,
                            self.profiler.text_hidden_size,
                        ),
                        mask=torch.ones(data_plan.bs, self.profiler.text_max_seq_len, dtype=torch.long),
                        num_frames=torch.tensor([data_plan.num_frame] * data_plan.bs),
                        height=torch.tensor([height] * data_plan.bs),
                        width=torch.tensor([width] * data_plan.bs),
                        fps=torch.tensor([24 if data_plan.num_frame > 1 else 120] * data_plan.bs),
                        ar=torch.tensor([height / width] * data_plan.bs),
                        plan_idx=data_idx,
                        warmup_iter=data_plan.warmup_iter,
                    )
                )
            yield ret

            if self.profiler.has_next_data_plan():
                self.data_plan.append(self.profiler.next_data_plan())
        self.profiler.finalize_profile()


class Profiler:
    def __init__(
        self,
        total_layers,
        bucket_config,
        text_max_seq_len,
        text_hidden_size,
        dynamic_sp,
        dynamic_recompute,
        auto_grad_acc,
        do_profile,
        distributed_profile,
        node_rank,
        node_size,
        alloc_fraction,
        sp_balance_scope="iter",
        profile_path=None,
        verbose=False,
        profile_depth=2,
        parallel_mgr=None,
    ):
        self.total_layers = total_layers

        # [(ar_name, num_frame)]
        self.bucket_config = []
        for ar_name in bucket_config:
            for num_frame in bucket_config[ar_name]:
                self.bucket_config.append((ar_name, num_frame))
        self.bucket_config = sorted(self.bucket_config, key=lambda x: ASPECT_RATIOS[x[0]][0] * x[1], reverse=True)

        self.text_max_seq_len = text_max_seq_len
        self.text_hidden_size = text_hidden_size

        self.dynamic_sp = dynamic_sp
        self.dynamic_recompute = dynamic_recompute
        self.auto_grad_acc = auto_grad_acc
        self.sp_balance_scope = sp_balance_scope
        self.do_profile = do_profile
        self.distributed_profile = distributed_profile
        self.node_rank = node_rank
        self.node_size = node_size
        self.alloc_fraction = alloc_fraction
        self.profile_path = profile_path
        self.parallel_mgr = parallel_mgr

        self.max_sp = torch.cuda.device_count()
        # in bytes
        self.memory_cap = alloc_fraction * torch.cuda.mem_get_info()[1]
        self.logger = logging if verbose else None
        self.profile_depth = profile_depth
        self.parallel_mgr = parallel_mgr

        self._load_profile()

        self.timers: Dict[str, GroupTimer] = dict()
        self.global_timer = None
        self.registered_timer_keys = []
        if self.need_profile():
            self.timers["iteration"] = GroupTimer("iteration")
        self.dummy_timer = nullcontext()

        # used by profile
        self.profile_unit_grad_in_bytes = 0
        self.module_dict = None

    ############################################################
    # init methods
    def _load_profile(self):
        self.profile_pbar = None
        if not self.do_profile:
            assert os.path.isdir(self.profile_path)
            self.profile_results = {}

            # Iterate through all profile_*.json files in the directory
            for filename in os.listdir(self.profile_path):
                if filename.startswith("profile") and filename.endswith(".json"):
                    profile_file = os.path.join(self.profile_path, filename)
                    with open(profile_file) as f:
                        partial_results = json.load(f)
                        # Merge results
                        for ar_name, num_frame_dict in partial_results.items():
                            if ar_name not in self.profile_results:
                                self.profile_results[ar_name] = {}
                            self.profile_results[ar_name].update(num_frame_dict)

            # Convert frame numbers from strings to integers
            for ar_name in self.profile_results:
                self.profile_results[ar_name] = {int(k): v for k, v in self.profile_results[ar_name].items()}

            if not self.dynamic_recompute and not self.auto_grad_acc:
                if not self.dynamic_sp or self.sp_balance_scope == "epoch":
                    self.interpolate_profile_results()

            self.next_bucket_idx = None
            self.next_sp_size = None
            self.next_bs = None
            self.next_warmup_iter = False

            self._need_profile = False
        else:
            self.profile_results = {}

            if self.distributed_profile:
                num_buckets = len(self.bucket_config)
                _div, _mod = divmod(num_buckets, self.node_size)
                cnts = [_div] * self.node_size
                for i in range(self.node_size - _mod, self.node_size):
                    cnts[i] += 1
                cnts = list(accumulate(cnts, initial=0))

                self.next_bucket_idx = cnts[self.node_rank]
                self.bucket_partition_boundary = cnts[self.node_rank + 1]
            else:
                self.next_bucket_idx = 0
                self.bucket_partition_boundary = len(self.bucket_config)

            self.next_sp_size = 1 if self.dynamic_sp else self.parallel_mgr.sp_size
            self.next_bs = 1
            self.next_warmup_iter = True

            self._need_profile = True

        self.profile_ctx = None
        self.latest_raw_result = None
        self.raw_results = []
        self.detail_results = []
        self.dp_results = []

        logging.info(f"Profile results: {pformat(self.profile_results, sort_dicts=False)}")

    def interpolate_profile_results(self):
        if not self.dynamic_recompute and not self.auto_grad_acc:
            if self.logger:
                self.logger.info(
                    f">>> [Profiling] Profile results before adjustment: {pformat(self.profile_results, sort_dicts=False)}"
                )

            def score_func(new_time, median_time):
                if not self.dynamic_sp:
                    return np.abs(new_time - median_time)
                if new_time > median_time:
                    return (new_time - median_time) * 1.5
                else:
                    return (median_time - new_time) * 1

            time_list = []
            for _, num_frame_dict in self.profile_results.items():
                for _, info in num_frame_dict.items():
                    time_list.append(info["max"]["execution_time"])
            median_time = np.median(time_list)

            for _, num_frame_dict in self.profile_results.items():
                for _, info in num_frame_dict.items():
                    cur_time = info["max"]["execution_time"]
                    cur_bs = info["max"]["bs"]
                    cur_diff = score_func(cur_time, median_time)
                    if cur_time > median_time:
                        if self.dynamic_sp and cur_bs == 1:
                            tmp_sp_size = info["sp_size"]
                            if tmp_sp_size == self.max_sp:
                                continue
                            while cur_time > median_time:
                                tmp_sp_size = tmp_sp_size * 2
                                cur_time = cur_time / 2
                                if tmp_sp_size > self.max_sp:
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

    ############################################################
    # status queries
    def need_profile(self):
        return self._need_profile

    def is_valid_bucket(self, ar_name, num_frame):
        return ar_name in self.profile_results and num_frame in self.profile_results[ar_name]

    def get_batch_size(self, ar_name, num_frame):
        return self.profile_results[ar_name][num_frame]["max"]["bs"]

    def get_sp_size(self, ar_name, num_frame):
        return self.profile_results[ar_name][num_frame]["sp_size"]

    def get_execution_time(self, ar_name, num_frame):
        return self.profile_results[ar_name][num_frame]["max"]["execution_time"]

    def get_recompute_cfg(self, ar_name, num_frame):
        return self.profile_results[ar_name][num_frame]["recompute_cfg"]

    ############################################################
    # Key functionality: profiling and planning for bs, sp size, and recompute cfg
    def get_data_iter(self):
        return ProfileDataIter(self)

    def has_next_data_plan(self):
        "Move to next bucket"
        return self.next_bucket_idx is not None and self.next_bucket_idx < self.bucket_partition_boundary

    def next_data_plan(self):
        ar_name, num_frame = self.bucket_config[self.next_bucket_idx]

        next_plan = DataPlan(
            ar_name=ar_name,
            num_frame=num_frame,
            sp_size=self.next_sp_size,
            gas=2 if self.auto_grad_acc else 1,
            bs=self.next_bs,
            warmup_iter=False if self.auto_grad_acc else self.next_warmup_iter,
        )

        return next_plan

    def update_next_data_plan(self):
        self.next_bucket_idx += 1
        self.profile_pbar.update()

        self.next_bs = 1
        self.next_warmup_iter = not self.auto_grad_acc

        if self.dynamic_sp:
            self.next_sp_size = 1

    def finalize_profile(self):
        assert self._need_profile
        self._need_profile = False
        self.profile_pbar.close()

        # exit profiling, dump results and clean up
        self.global_timer.__exit__(0, 0, 0)
        logging.info(f">>> [Profiling] Profile cost: {self.global_timer.elapsed_time:.2f} s")
        logging.info(f">>> [Profiling] Profile results: {pformat(self.profile_results, sort_dicts=False)}")

        if dist.get_rank() == 0:
            df = pd.DataFrame(
                [each.to_list() for each in self.raw_results],
                columns=["ar", "num_frame", "bs", "sp_size", "execution_time", "max_alloc_memory"]
                + self.profile_ctx.get_profile_fields(),
            )
            os.makedirs(self.profile_path, exist_ok=True)
            df.to_csv(f"{self.profile_path}/raw_results_{self.node_rank}-{self.node_size}.csv", index=False)

            detail_df = pd.DataFrame(
                self.detail_results,
                columns=["ar", "num_frame", "bs", "sp_size"]
                + ["pred_time", "pred_memory", "saved_time", "cost_memory"]
                + (
                    [f"{k}_cnt" for k in self.raw_results[0].submodule_fields]
                    if self.dynamic_recompute and len(self.raw_results) > 0
                    else []
                ),
            )
            detail_df.to_csv(f"{self.profile_path}/detail_profile_{self.node_rank}-{self.node_size}.csv", index=False)

            with open(f"{self.profile_path}/profile_{self.node_rank}-{self.node_size}.json", "w") as f:
                json.dump(self.profile_results, f)

        # reverse status
        torch.cuda.set_per_process_memory_fraction(1.0)
        self.profile_ctx = None
        disable_profile()
        self.global_timer = None
        clean_cache()

    def init_profiler(self):
        torch.cuda.set_per_process_memory_fraction(self.alloc_fraction)
        self.profile_pbar = tqdm(
            range(self.next_bucket_idx, self.bucket_partition_boundary),
            desc="Profiling",
            disable=dist.get_rank() != 0,
            initial=self.next_bucket_idx,
            total=self.bucket_partition_boundary - self.next_bucket_idx,
        )

        enable_profile(list(self.module_dict.keys()))
        self.profile_ctx = get_profile_context()

        self.global_timer = GroupTimer("global", group=self.parallel_mgr.dp_group)
        dist.barrier()
        self.global_timer.__enter__()

    def register_modules(self, module_dict):
        self.module_dict = module_dict
        for name, module_list in module_dict.items():
            assert len(module_list) == self.total_layers

            for module in module_list:
                setattr(module, "__profile_module_key", name)

            if self.auto_grad_acc:
                # only a single module
                for p in module_list[0].parameters():
                    if p.requires_grad:
                        self.profile_unit_grad_in_bytes += p.nbytes
        profile_unit_grad_in_bytes = torch.tensor([self.profile_unit_grad_in_bytes], device=torch.cuda.current_device())
        dist.all_reduce(profile_unit_grad_in_bytes, op=dist.ReduceOp.MAX)
        self.profile_unit_grad_in_bytes = int(profile_unit_grad_in_bytes.item())
        if self.logger:
            self.logger.info(
                f">>> [Profiling] Profile with grad accumulation, unit grad in bytes: {self.profile_unit_grad_in_bytes}"
            )

    @contextmanager
    def profile(self, batch, model, gas):
        if not self.need_profile():
            yield self.total_layers
            return

        ar_name = batch["ar_name"]
        num_frame = batch["num_frame"]
        sp_size = batch["sp_size"]
        batch_data = batch["data"][gas]
        bs = batch_data["video"].shape[0]

        plan_idx = batch_data.pop("plan_idx")
        warmup_iter = batch_data.pop("warmup_iter")
        if warmup_iter or (self.auto_grad_acc and gas == 0):
            clean_cache()

        if self.logger:
            self.get_memory_stats(
                f"START bucket {ar_name} {num_frame} {bs} with sp {sp_size} is wamrup: {warmup_iter}, gas: {gas}"
            )

        pass_depth_loop = True
        oom_error_types = (
            (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError)
            if hasattr(torch, "OutOfMemoryError")
            else torch.cuda.OutOfMemoryError
        )
        try:
            with self.timeit("iteration"):
                yield (self.total_layers if self.profile_depth is None else self.profile_depth)
            row = ProfileResult(
                ar_name,
                num_frame,
                bs,
                sp_size,
                execution_time=self.timers["iteration"].elapsed_time,
                max_alloc_memory=torch.cuda.max_memory_allocated(),
                input_memory=self.profile_ctx.get_input_memory(),
                submodule_fields=self.profile_ctx.get_submodule_fields(),
                submodule_fwd_time=self.profile_ctx.get_submodule_fwd_time(),
                submodule_bwd_time=self.profile_ctx.get_submodule_bwd_time(),
                submodule_memory=self.profile_ctx.get_submodule_memory(),
            )

        except oom_error_types as err_oom:
            if plan_idx == 0:
                print(f"unable to run the smallest video bucket in this hardware")
                raise err_oom
            reset_status(model)
            pass_depth_loop = False
            if self.logger:
                self.logger.info(
                    f">>> [Profiling] Bucket {ar_name} {num_frame} at {bs} sp {sp_size} doesn't pass profile, OOM!"
                )

        # warmup for lazy initialized optimizers like Adam(W)
        if plan_idx == 0:
            return

        # estimate for fast profiling
        is_success = False
        if pass_depth_loop:
            pred_full_time, pred_full_mem = self.estimate_overhead(row)

            if pred_full_mem <= self.memory_cap:
                is_success = True
            elif self.logger:
                self.logger.info(
                    f">>> [Profiling] Bucket {ar_name} {num_frame} at {bs} sp {sp_size} pass profile but exceed memory limit: {pred_full_mem/GB:.2f}/{self.memory_cap/GB:.2f} GB"
                )

        if (is_success and warmup_iter) or (self.auto_grad_acc and gas == 0):
            if self.next_warmup_iter:
                self.next_warmup_iter = False
            return

        if self.dynamic_sp:
            if is_success:
                self.raw_results.append(row)

                avail_mem = int(np.floor((self.memory_cap - pred_full_mem) / GB))

                if self.dynamic_recompute:
                    num_modules = len(row.submodule_fields)

                    dp = torch.zeros(avail_mem + 1, dtype=torch.float)
                    trace = torch.zeros((avail_mem + 1, num_modules), dtype=torch.int)

                    for i in range(num_modules):
                        module_time_cost = row.submodule_fwd_time[i]
                        module_mem_cost_in_bytes = row.submodule_memory[i]

                        temp_dp = dp.clone()
                        temp_trace = trace.clone()

                        for cnt in range(1, self.total_layers + 1):
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
                        avail_mem -= int(np.ceil(row.submodule_memory[i] * best_trace[i] / GB))
                    assert avail_mem >= 0, f"rest memory: {avail_mem}, trace: {best_trace}"
                    result_row = [
                        row.ar_name,
                        row.num_frame,
                        row.bs,
                        row.sp_size,
                        best_full_time,
                        self.memory_cap / GB - avail_mem,
                        reduced_time,
                        avail_mem,
                    ] + best_trace

                    self.dp_results.append(result_row)
                else:
                    result_row = [
                        row.ar_name,
                        row.num_frame,
                        row.bs,
                        row.sp_size,
                        pred_full_time,
                        pred_full_mem / GB,
                        0,
                        avail_mem,
                    ]

                    if self.auto_grad_acc:
                        self.dp_results.append(result_row)
                    else:
                        self.latest_raw_result = result_row
                
                self.detail_results.append(result_row)

                if self.logger:
                    self.logger.info(
                        f">>> [Profiling] DONE BS search for bucket {ar_name} {num_frame} at {bs} sp {sp_size}"
                    )

                self.next_bs = bs * 2
                self.next_warmup_iter = not self.auto_grad_acc
            else:
                if not self.dynamic_recompute and not self.auto_grad_acc:
                    if bs == 1:
                        if self.logger:
                            self.logger.info(
                                f">>> [Profiling] SKIP bucket {ar_name} {num_frame} which cannot fit into sp: {sp_size}"
                            )
                    else:
                        assert self.latest_raw_result is not None
                        self.dp_results.append(self.latest_raw_result)
                    
                if sp_size < self.max_sp:
                    self.next_sp_size = sp_size * 2
                    # bs // 2 is the previous successful bs
                    self.next_bs = 1 if self.dynamic_recompute else max(1, bs//4)
                    self.next_warmup_iter = not self.auto_grad_acc
                elif len(self.dp_results) == 0:
                    if self.logger:
                        self.logger.info(
                            f">>> [Profiling] SKIP bucket {ar_name} {num_frame} which cannot fit into the cluster"
                        )
                    self.update_next_data_plan()
                else:
                    if self.logger:
                        self.logger.info(
                            f">>> [Profiling] STOP BS search for bucket {ar_name} {num_frame} at {bs} sp {sp_size}"
                        )

                    best = sorted(self.dp_results, key=lambda x: x[2] / x[3] / x[4], reverse=True)[0]

                    ar_name, num_frame, bs, sp_size, execution_time, memory_consumed = best[:6]

                    if ar_name not in self.profile_results:
                        self.profile_results[ar_name] = {}
                    self.profile_results[ar_name][num_frame] = dict(
                        sp_size=sp_size,
                        max=dict(
                            bs=bs,
                            execution_time=execution_time,
                            memory_consumed=memory_consumed,
                        ),
                    )

                    if self.dynamic_recompute:
                        self.profile_results[ar_name][num_frame]["recompute_cfg"] = {
                            k: best[i + 8] for i, k in enumerate(list(self.module_dict.keys()))
                        }

                    self.dp_results = []
                    self.update_next_data_plan()

        else:  # baseline
            if is_success:
                # non warm up iter, record this result, increase bs
                self.latest_raw_result = row

                self.next_bs *= 2
                self.next_warmup_iter = True
            else:
                if bs == 1:
                    if self.logger is not None:
                        self.logger.info(
                            f">>> [Profiling] SKIP bucket {ar_name} {num_frame} which cannot fit into the cluster"
                        )
                else:
                    assert self.latest_raw_result is not None
                    if self.logger:
                        self.logger.info(
                            f">>> [Profiling] STOP BS search for bucket {ar_name} {num_frame} at {bs} sp {sp_size}"
                        )

                    ar_name = self.latest_raw_result.ar_name
                    num_frame = self.latest_raw_result.num_frame
                    bs = self.latest_raw_result.bs
                    sp_size = self.latest_raw_result.sp_size

                    pred_full_time, pred_full_mem = self.estimate_overhead(self.latest_raw_result)

                    if ar_name not in self.profile_results:
                        self.profile_results[ar_name] = {}
                    self.profile_results[ar_name][num_frame] = dict(
                        sp_size=sp_size,
                        max=dict(
                            bs=bs,
                            execution_time=pred_full_time,
                            memory_consumed=pred_full_mem / GB,
                        ),
                    )
                    self.raw_results.append(self.latest_raw_result)
                    self.latest_raw_result = None

                self.update_next_data_plan()

    def get_memory_stats(self, phase=None):
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

    def estimate_overhead(self, profile_result: ProfileResult):
        fwd_time = sum(profile_result.submodule_fwd_time)
        bwd_time = sum(profile_result.submodule_bwd_time)

        layer_time = fwd_time * 2 + bwd_time
        layer_memory = sum(profile_result.input_memory)

        pred_full_time = profile_result.execution_time + layer_time * (self.total_layers - self.profile_depth)
        pred_full_mem = profile_result.max_alloc_memory + (layer_memory + self.profile_unit_grad_in_bytes) * (
            self.total_layers - self.profile_depth
        )

        return pred_full_time, pred_full_mem

    def optimize_dynamics(self, batch, model):
        # set sequence parallel size if args.dynamic_sp is True
        sp_size = batch["sp_size"]
        if isinstance(self.parallel_mgr, DynamicParallelManager):
            self.parallel_mgr.set_sp_size(sp_size)
        self.update_timer_group()

        # set grad accumulation steps if args.auto_grad_accumulation is True
        total_gas = batch["gas"]
        if batch.get("profile_grad_acc", False):
            plan_idx = batch["data"][-1]["plan_idx"]
            if plan_idx == 1:
                set_grad_accumulation_steps(model, 1000000)
        else:
            set_grad_accumulation_steps(model, total_gas)

        # set recompute cfg if args.dynamic_recompute is True
        ar_name = batch["ar_name"]
        num_frame = batch["num_frame"]
        if self.dynamic_recompute and not self.need_profile():
            recompute_cfg = self.get_recompute_cfg(ar_name, num_frame)
            for d in range(self.total_layers):
                for k, module_list in self.module_dict.items():
                    module_list[d].grad_checkpointing = d >= recompute_cfg[k]

    def set_gradient_accumulation_boundary(self, model, batch, gas):
        """
        a hack to enable automatic gradient accumulation with deepspeed engine

        is there any workaround to remove this line?
        """
        total_gas = batch["gas"]
        if batch.get("profile_grad_acc", False):
            model.set_gradient_accumulation_boundary(False)
        else:
            model.set_gradient_accumulation_boundary(gas == total_gas - 1)

    ############################################################
    # Functionality: timing. Refer to args.register_timer_keys and train_step

    @contextmanager
    def timeit(self, name):
        if name in self.timers:
            timer = self.timers[name]
        else:
            timer = self.dummy_timer

        with timer:
            yield

    def update_timer_group(self):
        if self.need_profile():
            cur_group = self.parallel_mgr.sp_group
            for t in self.timers:
                self.timers[t].group = cur_group


def set_profiler(
    total_layers,
    bucket_config,
    text_max_seq_len,
    text_hidden_size,
    dynamic_sp,
    dynamic_recompute,
    auto_grad_acc,
    do_profile,
    distributed_profile,
    node_rank,
    node_size,
    alloc_fraction,
    profile_path,
    parallel_mgr,
    verbose,
) -> Profiler:
    global PROFILER
    PROFILER = Profiler(
        total_layers=total_layers,
        bucket_config=bucket_config,
        text_max_seq_len=text_max_seq_len,
        text_hidden_size=text_hidden_size,
        dynamic_sp=dynamic_sp,
        dynamic_recompute=dynamic_recompute,
        auto_grad_acc=auto_grad_acc,
        do_profile=do_profile,
        distributed_profile=distributed_profile,
        node_rank=node_rank,
        node_size=node_size,
        alloc_fraction=alloc_fraction,
        profile_path=profile_path,
        parallel_mgr=parallel_mgr,
        verbose=verbose,
    )
    return PROFILER


def get_profiler():
    global PROFILER
    return PROFILER
