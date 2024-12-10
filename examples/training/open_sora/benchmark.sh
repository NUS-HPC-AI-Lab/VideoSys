# =============== environment variables ================
export PYTHONPATH=$PYTHONPATH:$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# =============== benchmark commands ================
# If you run this for the first time, you need to run the program twice.
# The first time is to profile the model and save results, and the second time is to run the benchmark.

# baseline
torchrun --standalone --nproc_per_node 8 examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/baseline.yaml

# DCP intra
torchrun --standalone --nproc_per_node 8 examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/dcp_intra.yaml

# DCP inter
torchrun --standalone --nproc_per_node 8 examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/dcp_inter.yaml

# DCP inter + ckpt
torchrun --standalone --nproc_per_node 8 examples/training/open_sora/train.py \
    examples/training/open_sora/configs/benchmarks/dcp_inter_ckpt.yaml
