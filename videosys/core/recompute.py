import time
from enum import Enum, auto

import torch
from torch.utils.checkpoint import checkpoint

PROFILE_CONTEXT = None


class STDiT3BlockRecomputeConfig(Enum):
    NONE = auto()
    BLOCK = auto()
    SELF_ATTN = auto()
    CROSS_ATTN = auto()
    MLP = auto()
    SELF_AND_CROSS_ATTN = auto()
    CROSS_ATTN_AND_MLP = auto()
    SELF_ATTN_AND_MLP = auto()


class SpatialTemporalProfileContext:
    """
    This profiler context should only be used within sampler profile process.
    """

    def __init__(
        self,
    ):
        self.keys = [
            "layer",
        ]
        self.prev_time = 0
        self.prev_memory = 0

        self.fwd_time_record = dict()
        self.bwd_time_record = dict()
        self.memory_record = dict()
        self.submodule_fields = []

        self.memory_record["spatial_input_memory"] = 0
        self.memory_record["temporal_input_memory"] = 0
        for prefix in ["spatial_", "temporal_"]:
            for key in self.keys:
                self.fwd_time_record[prefix + key + "_fwd"] = 0.0
                self.bwd_time_record[prefix + key + "_bwd"] = 0.0
                self.memory_record[prefix + key + "_memory"] = 0
                self.submodule_fields.append(prefix + key)

    def record(self, prefix, tick, time_stamp, memory, tensor_nbytes, is_fwd):
        key = "null"
        if is_fwd:
            if tick == 0:
                key = prefix + "input_memory"
                self.memory_record[key] = tensor_nbytes
            else:
                key = prefix + self.keys[tick - 1]
                self.fwd_time_record[key + "_fwd"] = time_stamp - self.prev_time
                self.memory_record[key + "_memory"] = memory - self.prev_memory
            self.prev_memory = memory
        else:
            if tick < len(self.keys):
                key = prefix + self.keys[tick]
                self.bwd_time_record[key + "_bwd"] = time_stamp - self.prev_time

        self.prev_time = time_stamp
        # print(f"rank: {torch.distributed.get_rank()} key: {key}, is fwd: {is_fwd}"
        #       f", time: {self.prev_time}, memory: {self.prev_memory/1024**2}")

    def get_profile_fields(self):
        return list(self.fwd_time_record.keys()) + list(self.bwd_time_record.keys()) + list(self.memory_record.keys())

    def get_profile_results(self):
        return (
            list(self.fwd_time_record.values())
            + list(self.bwd_time_record.values())
            + list(self.memory_record.values())
        )

    def get_submodule_fields(self):
        return self.submodule_fields

    def get_submodule_fwd_time(self):
        return list(self.fwd_time_record.values())

    def get_submodule_bwd_time(self):
        return list(self.bwd_time_record.values())

    def get_submodule_memory(self):
        return [self.memory_record[each + "_memory"] for each in self.submodule_fields]

    def get_input_memory(self):
        return [self.memory_record["spatial_input_memory"], self.memory_record["temporal_input_memory"]]


class BlockProfileContext:
    def __init__(self):
        self.prev_time = 0
        self.prev_memory = 0

        self.fwd_time_record = 0
        self.bwd_time_record = 0
        self.input_memory = 0
        self.layer_memory = 0

    def record(self, tick, time_stamp, memory, tensor_nbytes, is_fwd):
        if is_fwd:
            if tick == 0:
                self.input_memory = tensor_nbytes
            else:
                self.fwd_time_record = time_stamp - self.prev_time
                self.layer_memory = memory - self.prev_memory
            self.prev_memory = memory
        else:
            if tick == 0:
                self.bwd_time_record = time_stamp - self.prev_time
        self.prev_time = time_stamp

    def get_profile_fields(self):
        return ["layer_fwd", "layer_bwd", "input_memory", "layer_memory"]

    def get_profile_results(self):
        return [self.fwd_time_record, self.bwd_time_record, self.input_memory, self.layer_memory]

    def get_submodule_fields(self):
        return ["layer"]

    def get_submodule_fwd_time(self):
        return [self.fwd_time_record]

    def get_submodule_bwd_time(self):
        return [self.bwd_time_record]

    def get_submodule_memory(self):
        return [self.layer_memory]

    def get_input_memory(self):
        return [self.input_memory]


class TimeStamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, tick):
        torch.cuda.synchronize()
        time_stamp = time.time()
        memory = torch.cuda.memory_allocated()

        global PROFILE_CONTEXT
        PROFILE_CONTEXT.record(tick, time_stamp, memory, tensor.nbytes, True)

        ctx.tick = tick
        return tensor

    @staticmethod
    def backward(ctx, grad_output):
        torch.cuda.synchronize()
        time_stamp = time.time()
        memory = torch.cuda.memory_allocated()

        global PROFILE_CONTEXT
        PROFILE_CONTEXT.record(ctx.tick, time_stamp, memory, 0, False)
        return grad_output, None, None


def add_timestamp(tensor, tick):
    tensor = TimeStamp.apply(tensor, tick)
    return tensor


def enable_profile():
    global PROFILE_CONTEXT
    PROFILE_CONTEXT = BlockProfileContext()


def disable_profile():
    global PROFILE_CONTEXT
    PROFILE_CONTEXT = None


def get_profile_context():
    global PROFILE_CONTEXT
    return PROFILE_CONTEXT


def recompute_func(forward_func, depth, x, *args, **kwargs):
    x = add_timestamp(x, 0)
    x = forward_func(depth, x, *args, **kwargs)
    x = add_timestamp(x, 1)
    return x


def auto_recompute(model_config, forward_func, depth, x, *args, **kwargs):
    global PROFILE_CONTEXT
    if_enable_profile = PROFILE_CONTEXT is not None

    if if_enable_profile:
        output = checkpoint(recompute_func, forward_func, depth, x, *args, **kwargs, use_reentrant=True)
    elif depth < model_config.non_recompute_depth:
        output = forward_func(depth, x, *args, **kwargs)
    else:
        output = checkpoint(forward_func, depth, x, *args, **kwargs, use_reentrant=False)

    return output
