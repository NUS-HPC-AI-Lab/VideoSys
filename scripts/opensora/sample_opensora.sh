export CUDA_VISIBLE_DEVICES=6
torchrun --standalone --nproc_per_node=1 scripts/opensora/sample_opensora.py --config configs/opensora/sample.yaml

# skip
export CUDA_VISIBLE_DEVICES=1,2,5,7
torchrun --standalone --nproc_per_node=4 scripts/opensora/sample_opensora.py --config configs/opensora/sample_skip.yaml
