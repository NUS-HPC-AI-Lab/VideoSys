CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nproc_per_node=1 scripts/latte/sample_latte.py --config configs/latte/sample.yaml > sample_latte_baseline.txt 2>&1 &

CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nproc_per_node=1 scripts/latte/sample_latte.py --config configs/latte/sample_skip.yaml > sample_latte_skip.txt 2>&1 &
