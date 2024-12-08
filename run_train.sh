# 这个程序需要四台机器，每台机器八卡，每台机器都要运行这个脚本
# 需要改的地方：
# 1. node rank， 第13行
# 2. mask addr， 第17行

# =============== program params ================
export PYTHONPATH=$PYTHONPATH:$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

export NNODES=4 # 四台机器
# node rank就是第几台机器，需要自己改一下
export NODE_RANK=
export GPUS_PER_NODE=8
export WORLD_SIZE=$(($NNODES*$GPUS_PER_NODE))
# mask addr就是第一台机器的ip地址
export MASTER_ADDR=
export MASTER_PORT=9527


# =============== baseline ================
torchrun \
    --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/baseline.yaml \
    --image-mixing-frac 1 > out_train_zipf1_baseline.txt 2>&1

# =============== dynamic sp ================
torchrun \
    --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/dcp_intra.yaml \
    --image-mixing-frac 1 > out_train_zipf1_dcp_intra.txt 2>&1

# =============== dynamic sp + ga ================
torchrun \
    --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/dcp_inter.yaml \
    --image-mixing-frac 1 > out_train_zipf1_dcp_inter.txt 2>&1

# =============== dynamic sp + ckpt + ga ================
torchrun \
    --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/dcp_inter_ckpt.yaml \
    --image-mixing-frac 1 > out_train_zipf1_dcp_inter_ckpt.txt 2>&1

# =============== baseline ================
torchrun \
    --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/baseline.yaml \
    --image-mixing-frac 2 > out_train_zipf2_baseline.txt 2>&1

# =============== dynamic sp ================
torchrun \
    --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/dcp_intra.yaml \
    --image-mixing-frac 2 > out_train_zipf2_dcp_intra.txt 2>&1

# =============== dynamic sp + ga ================
torchrun \
    --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/dcp_inter.yaml \
    --image-mixing-frac 2 > out_train_zipf2_dcp_inter.txt 2>&1

# =============== dynamic sp + ckpt + ga ================
torchrun \
    --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/dcp_inter_ckpt.yaml \
    --image-mixing-frac 2 > out_train_zipf2_dcp_inter_ckpt.txt 2>&1

# =============== baseline ================
torchrun \
    --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/baseline.yaml \
    --image-mixing-frac 5 > out_train_zipf5_baseline.txt 2>&1

# =============== dynamic sp ================
torchrun \
    --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/dcp_intra.yaml \
    --image-mixing-frac 5 > out_train_zipf5_dcp_intra.txt 2>&1

# =============== dynamic sp + ga ================
torchrun \
    --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/dcp_inter.yaml \
    --image-mixing-frac 5 > out_train_zipf5_dcp_inter.txt 2>&1

# =============== dynamic sp + ckpt + ga ================
torchrun \
    --nproc_per_node=$GPUS_PER_NODE --nnodes=$NNODES --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/dcp_inter_ckpt.yaml \
    --image-mixing-frac 5 > out_train_zipf5_dcp_inter_ckpt.txt 2>&1
