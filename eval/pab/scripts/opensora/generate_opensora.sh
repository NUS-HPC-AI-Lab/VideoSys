# origin
torchrun --standalone --nproc_per_node=8 evaluations/fastvideodiffusion/scripts/opensora/generate_opensora.py --config evaluations/fastvideodiffusion/configs/opensora/sample.yaml

# fvd
torchrun --standalone --nproc_per_node=8 evaluations/fastvideodiffusion/scripts/opensora/generate_opensora.py --config evaluations/fastvideodiffusion/configs/opensora/sample_pab.yaml
