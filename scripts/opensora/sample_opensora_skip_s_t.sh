
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 scripts/opensora/sample_opensora_skip_s_t.py --config configs/opensora/sample_skip_s_t_1.yaml > sample_skip_s_t_1.txt 2>&1

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 scripts/opensora/sample_opensora_skip_s_t.py --config configs/opensora/sample_skip_s_t_2.yaml > sample_skip_s_t_2.txt 2>&1

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 scripts/opensora/sample_opensora_skip_s_t.py --config configs/opensora/sample_skip_s_t_3.yaml > sample_skip_s_t_3.txt 2>&1

CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc_per_node=1 scripts/opensora/sample_opensora_skip_s_t.py --config configs/opensora/sample_skip_s_t_4.yaml > sample_skip_s_t_4.txt 2>&1
