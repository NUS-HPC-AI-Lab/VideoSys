CUDA_VISIBLE_DEVICES=7 torchrun --standalone --nproc_per_node=1 scripts/opensora/sample_opensora_mse.py --config configs/opensora/sample_skip.yaml > sample_opensora_mse.txt 2>&1 &
