# =============== program params ================
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# =============== preprocess ===============
torchrun --standalone --nproc_per_node 8 examples/training/open_sora/preprocess.py examples/training/open_sora/configs/preprocess.yaml

# =============== train ===============
# If you run this for the first time, you need to run the program twice.
# The first time is to profile the model and save results, and the second time is to run the benchmark.
torchrun --standalone --nproc_per_node 8 examples/training/open_sora/train.py examples/training/open_sora/configs/train.yaml
