CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nproc_per_node=1 scripts/opensora/sample_opensora.py --config configs/opensora/sample_skip.yaml > log/opensora_sample_skip.txt 2>&1 &

CUDA_VISIBLE_DEVICES=7 torchrun --standalone --nproc_per_node=1 scripts/opensora/sample_opensora.py --config configs/opensora/sample_skip_s_t.yaml > log/opensora_sample_skip_s_t.txt 2>&1 &


# skip
# torchrun --standalone --nproc_per_node=8 scripts/opensora/sample_opensora.py --config configs/opensora/sample_skip.yaml
