CUDA_VISIBLE_DEVICES=7 torchrun --standalone --nproc_per_node=1 scripts/latte/sample_latte_mse.py --config configs/latte/sample_skip.yaml > sample_latte_mse.txt 2>&1 &
