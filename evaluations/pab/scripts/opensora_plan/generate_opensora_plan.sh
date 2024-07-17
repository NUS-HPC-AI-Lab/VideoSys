# origin
torchrun --standalone --nproc_per_node=8 evaluations/fastvideodiffusion/scripts/opensora_plan/generate_opensora_plan.py --config evaluations/fastvideodiffusion/configs/opensora_plan/sample_65f.yaml

# pab
torchrun --standalone --nproc_per_node=8 evaluations/fastvideodiffusion/scripts/opensora_plan/generate_opensora_plan.py --config evaluations/fastvideodiffusion/configs/opensora_plan/sample_65f_pab.yaml
