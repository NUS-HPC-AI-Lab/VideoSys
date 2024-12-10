# =============== environment variables ================
# used by torch.distributed
export NNODES=1
export NODE_RANK=0
export GPUS_PER_NODE=8
export WORLD_SIZE=$(($NNODES*$GPUS_PER_NODE))
export MASTER_ADDR="localhost"
export MASTER_PORT=29502

export PYTHONPATH=$PYTHONPATH:$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# =============== launch commands ================
# baseline
torchrun --standalone --nproc_per_node 8 examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/baseline.yaml > baseline.log 2>& 1

# DCP intra
torchrun --standalone --nproc_per_node 8 examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/dcp_intra.yaml > dcp_intra.log 2>& 1

# DCP inter
torchrun --standalone --nproc_per_node 8 examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/dcp_inter.yaml > dcp_inter.log 2>& 1

# DCP inter + ckpt
torchrun --standalone --nproc_per_node 8 examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/dcp_inter_ckpt.yaml > dcp_inter_ckpt.log 2>& 1
