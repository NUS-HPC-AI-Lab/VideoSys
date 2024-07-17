CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 scripts/opensora_plan/sample_opensora_plan.py --config configs/opensora_plan/sample_65f_skip.yaml > log/opensora_plan_sample_65f_skip.txt 2>&1

CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 scripts/opensora_plan/sample_opensora_plan.py --config configs/opensora_plan/sample_65f_skip_s_t.yaml > log/opensora_plan_sample_65f_skip_s_t.txt 2>&1


# torchrun --standalone --nproc_per_node=8 scripts/opensora_plan/sample_opensora_plan.py --config configs/opensora_plan/sample_65f_skip.yaml
