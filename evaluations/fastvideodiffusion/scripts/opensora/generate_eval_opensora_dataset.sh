torchrun --standalone --nproc_per_node=1 evaluations/fastvideodiffusion/scripts/opensora/sample_opensora.py --config evaluations/fastvideodiffusion/configs/opensora/sample.yaml

# skip
# torchrun --standalone --nproc_per_node=8 evaluations/fastvideodiffusion/scripts/opensora/sample_opensora.py --config evaluations/fastvideodiffusion/configs/opensora/sample_skip.yaml