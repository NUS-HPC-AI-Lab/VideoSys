export CUDA_VISIBLE_DEVICES=6
torchrun --standalone --nproc_per_node=1 scripts/opensora/sample_opensora.py --config configs/opensora/sample.yaml

# skip
export CUDA_VISIBLE_DEVICES=3
torchrun --standalone --nproc_per_node=1 scripts/opensora/sample_opensora.py --config configs/opensora/sample_skip.yaml
