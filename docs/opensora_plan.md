# Open-Sora-Plan

## Inference

OpenDiT currently supports Open-Sora-Plan inference.

Usage:

```
bash scripts/opensora_plan/sample_opensora_plan.sh
```

### Low-Latency Inference with [PAB](./docs/pab.md)

You can perform low-latency video inference using our Pyramid Attention Broadcast (PAB) as follows.

```shell
torchrun --standalone --nproc_per_node=8 scripts/opensora_plan/sample_opensora_plan.py --config configs/opensora_plan/sample_65f_skip.yaml
```
