
CUDA_VISIBLE_DEVICES=7 torchrun --standalone --nproc_per_node=1 scripts/opensora/sample_opensora_skip.py --config configs/opensora/sample_skip_6.yaml > log_sample_skip_6.txt 2>&1

CUDA_VISIBLE_DEVICES=7 torchrun --standalone --nproc_per_node=1 scripts/opensora/sample_opensora_skip.py --config configs/opensora/sample_skip_7.yaml > log_sample_skip_7.txt 2>&1

CUDA_VISIBLE_DEVICES=7 torchrun --standalone --nproc_per_node=1 scripts/opensora/sample_opensora_skip.py --config configs/opensora/sample_skip_8.yaml > log_sample_skip_8.txt 2>&1

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 scripts/opensora/sample_opensora_skip.py --config configs/opensora/sample_skip.yaml > log_sample_skip.txt 2>&1
