torchrun --standalone --nproc_per_node=1 scripts/latte/sample.py --config configs/latte/sample.yaml

# Dynamic Sequence Parallelism
# torchrun --standalone --nproc_per_node=8 scripts/latte/sample.py --config configs/latte/sample.yaml

# Pyramidal Attention Broadcast
# torchrun --standalone --nproc_per_node=8 scripts/latte/sample.py --config configs/latte/sample_pab.yaml
