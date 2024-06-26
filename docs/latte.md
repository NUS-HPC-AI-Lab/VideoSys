# Latte

OpenDiT currently supports Latte basic inference.

Usage:

```
bash scripts/latte/sample_latte.sh
```


### Low-Latency Inference with [PAB](./docs/pab.md)

You can perform low-latency video inference using our Pyramid Attention Broadcast (PAB) as follows.

```shell
torchrun --standalone --nproc_per_node=8 scripts/latte/sample_latte.py --config configs/latte/sample_skip.yaml
```
