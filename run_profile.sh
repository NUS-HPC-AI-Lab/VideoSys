# 这个程序八卡跑就可以
# 不用改

# =============== program params ================
export PYTHONPATH=$PYTHONPATH:$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# =============== baseline ================
torchrun --standalone --nproc_per_node 4 examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/baseline.yaml > out_profile_baseline.txt 2>&1

# =============== dynamic sp ================
torchrun --standalone --nproc_per_node 8 examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/dcp_intra.yaml > out_profile_dcp_intra.txt 2>&1

# =============== dynamic sp + ga ================
torchrun --standalone --nproc_per_node 8 examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/dcp_inter.yaml > out_profile_dcp_inter.txt 2>&1

# =============== dynamic sp + ckpt + ga ================
torchrun --standalone --nproc_per_node 8 examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/dcp_inter_ckpt.yaml > out_profile_dcp_inter_ckpt.txt 2>&1
