import time
from enum import Enum, auto
import logging

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


class ProfileContext:
    """
    This profiler context should only be used within profiler.profile process.
    """

    def __init__(
        self,
        module_keys,
    ):
        self.module_keys = module_keys
        logging.info(f"module keys: {module_keys}")

        self.prev_time = 0
        self.prev_memory = 0

        self.fwd_time_record = dict()
        self.bwd_time_record = dict()
        self.input_memory = dict()
        self.layer_memory = dict()
        
        for k in self.module_keys:
            self.fwd_time_record[k] = 0
            self.bwd_time_record[k] = 0
            self.input_memory[k] = 0
            self.layer_memory[k] = 0

    def record(self, module_name, tick, time_stamp, memory, input_memory_in_bytes, is_fwd):
        if is_fwd:
            if tick == 0:
                self.input_memory[module_name] = input_memory_in_bytes
            else:
                self.fwd_time_record[module_name] = time_stamp - self.prev_time
                self.layer_memory[module_name] = memory - self.prev_memory
            self.prev_memory = memory
        else:
            if tick == 0:
                self.bwd_time_record[module_name] = time_stamp - self.prev_time
        self.prev_time = time_stamp

        # print(f"rank: {torch.distributed.get_rank()} key: {key}, is fwd: {is_fwd}"
        #       f", time: {self.prev_time}, memory: {self.prev_memory/1024**2}")

    def get_profile_fields(self):
        fields = []
        for k in self.module_keys:
            fields.append(k + "_fwd")
            fields.append(k + "_bwd")
            fields.append(k + "_input_memory")
            fields.append(k + "_layer_memory")
        return fields

    def get_submodule_fields(self):
        return self.module_keys

    def get_submodule_fwd_time(self):
        return list(self.fwd_time_record.values())

    def get_submodule_bwd_time(self):
        return list(self.bwd_time_record.values())

    def get_submodule_memory(self):
        return list(self.layer_memory.values())

    def get_input_memory(self):
        return list(self.input_memory.values())


class TimeStamp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, module_key, tick, *inputs):
        torch.cuda.synchronize()
        time_stamp = time.time()
        memory = torch.cuda.memory_allocated()

        input_memory_in_bytes = sum([x.nbytes for x in inputs if (isinstance(x, torch.Tensor) and x.requires_grad)])

        global PROFILE_CONTEXT
        PROFILE_CONTEXT.record(module_key, tick, time_stamp, memory, input_memory_in_bytes, True)

        ctx.prefix = module_key
        ctx.tick = tick
        return inputs

    @staticmethod
    def backward(ctx, *grad_output):
        torch.cuda.synchronize()
        time_stamp = time.time()
        memory = torch.cuda.memory_allocated()

        global PROFILE_CONTEXT
        PROFILE_CONTEXT.record(ctx.prefix, ctx.tick, time_stamp, memory, 0, False)
        return (None, None) + grad_output


def add_timestamp(module_key, *inputs, tick):
    inputs = TimeStamp.apply(module_key, tick, *inputs)
    return inputs


def enable_profile(module_keys):
    global PROFILE_CONTEXT
    PROFILE_CONTEXT = ProfileContext(module_keys)


def disable_profile():
    global PROFILE_CONTEXT
    PROFILE_CONTEXT = None


def get_profile_context():
    global PROFILE_CONTEXT
    return PROFILE_CONTEXT


def recompute_func(module, *args):
    args = add_timestamp(module.__profile_module_key, *args, tick=0)    
    args = module(*args)
    
    ret_as_tuple = True
    if not isinstance(args, tuple):
        ret_as_tuple = False
        args = (args,)
    args = add_timestamp(module.__profile_module_key, *args, tick=1)
    if ret_as_tuple:
        return args
    return args[0]


def auto_recompute(block_module, *args):
    global PROFILE_CONTEXT
    if_enable_profile = PROFILE_CONTEXT is not None

    if if_enable_profile:
        assert block_module.grad_checkpointing == True
        output = checkpoint(recompute_func, block_module, *args, use_reentrant=True)
    elif block_module.grad_checkpointing:
        output = checkpoint(block_module, *args, use_reentrant=False)
    else:
        output = block_module(*args)

    return output
