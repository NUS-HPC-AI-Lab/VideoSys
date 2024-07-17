CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 scripts/latte/sample_latte.py --config configs/latte/sample_skip.yaml > log/latte_sample_skip.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 scripts/latte/sample_latte.py --config configs/latte/sample_skip_s_t.yaml > log/latte_sample_skip_s_t.txt 2>&1 &
