
CUDA_VISIBLE_DEVICES=4 torchrun --standalone --nproc_per_node=1 scripts/opensora_plan/sample_opensora_plan_skip_s_t.py --config configs/opensora_plan/sample_65f_skip_s_t_1.yaml > opensora_plan_sample_65f_skip_s_t_1.txt 2>&1 &
echo "Finished opensora_plan_sample_65f_skip_s_t_1.txt"

CUDA_VISIBLE_DEVICES=5 torchrun --standalone --nproc_per_node=1 scripts/opensora_plan/sample_opensora_plan_skip_s_t.py --config configs/opensora_plan/sample_65f_skip_s_t_2.yaml > opensora_plan_sample_65f_skip_s_t_2.txt 2>&1 &
echo "Finished opensora_plan_sample_65f_skip_s_t_2.txt"

CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nproc_per_node=1 scripts/opensora_plan/sample_opensora_plan_skip_s_t.py --config configs/opensora_plan/sample_65f_skip_s_t_3.yaml > opensora_plan_sample_65f_skip_s_t_3.txt 2>&1 &
echo "Finished opensora_plan_sample_65f_skip_s_t_3.txt"

CUDA_VISIBLE_DEVICES=7 torchrun --standalone --nproc_per_node=1 scripts/opensora_plan/sample_opensora_plan_skip_s_t.py --config configs/opensora_plan/sample_65f_skip_s_t_4.yaml > opensora_plan_sample_65f_skip_s_t_4.txt 2>&1 &
echo "Finished opensora_plan_sample_65f_skip_s_t_4.txt"

CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nproc_per_node=1 scripts/opensora_plan/sample_opensora_plan_skip_s_t.py --config configs/opensora_plan/sample_65f_skip_s_t_5.yaml > opensora_plan_sample_65f_skip_s_t_5.txt 2>&1 &
echo "Finished opensora_plan_sample_65f_skip_s_t_3.txt"

CUDA_VISIBLE_DEVICES=7 torchrun --standalone --nproc_per_node=1 scripts/opensora_plan/sample_opensora_plan_skip_s_t.py --config configs/opensora_plan/sample_65f_skip_s_t_6.yaml > opensora_plan_sample_65f_skip_s_t_6.txt 2>&1 &
echo "Finished opensora_plan_sample_65f_skip_s_t_4.txt"
