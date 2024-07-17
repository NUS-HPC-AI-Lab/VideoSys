CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 scripts/latte/sample_latte_skip_s_t.py --config configs/latte/sample_skip_s_t_1.yaml > sample_skip_s_t_1.txt 2>&1 &

CUDA_VISIBLE_DEVICES=1 torchrun --standalone --nproc_per_node=1 scripts/latte/sample_latte_skip_s_t.py --config configs/latte/sample_skip_s_t_2.yaml > sample_skip_s_t_2.txt 2>&1 &

CUDA_VISIBLE_DEVICES=7 torchrun --standalone --nproc_per_node=1 scripts/latte/sample_latte_skip_s_t.py --config configs/latte/sample_skip_s_t.yaml > sample_skip_s_t_17.txt 2>&1 &
