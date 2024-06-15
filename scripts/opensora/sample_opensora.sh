torchrun --standalone --nproc_per_node=1 scripts/opensora/sample_opensora.py --config configs/opensora/sample.yaml

# speedup
# torchrun --standalone --nproc_per_node=8 scripts/opensora/sample_opensora.py --config configs/opensora/sample_speedup.yaml
