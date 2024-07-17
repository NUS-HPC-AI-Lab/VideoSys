# origin
torchrun --standalone --nproc_per_node=8 evaluations/fastvideodiffusion/scripts/latte/generate_latte.py --config evaluations/fastvideodiffusion/configs/latte/sample.yaml

# fvd
torchrun --standalone --nproc_per_node=8 evaluations/fastvideodiffusion/scripts/latte/generate_latte.py --config evaluations/fastvideodiffusion/configs/latte/sample_pab.yaml
