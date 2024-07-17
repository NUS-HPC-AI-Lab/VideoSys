# Latte

Latte: Latent Diffusion Transformer for Video Generation

## Inference

You can perform video inference for Latte-1 as follows.

```shell
# Use script
bash scripts/latte/sample.sh
# Use command line
torchrun --standalone --nproc_per_node=1 scripts/latte/sample.py --config configs/latte/sample.yaml
```

You can change settings in yaml config. Here are details of some key arguments:

- `--nproc_per_node`: The GPU number you want to use for the current node. Multiple GPUs inference will enable Dynamic Sequence Parallel.

### Inference with [PAB](./docs/pab.md)

PAB provides more efficient inference at the cost of minor quality loss. You can run as follows:

```shell
# Use script
bash scripts/latte/sample_pab.sh
# Use command line
torchrun --standalone --nproc_per_node=8 scripts/latte/sample.py --config configs/latte/sample_pab.yaml
```

You can change settings in yaml config. Here are details of some key arguments for training:

- `--nproc_per_node`: The GPU number you want to use for the current node. Multiple GPUs inference will enable Dynamic Sequence Parallel.

For the more detailed args and usages related with PAB, please refer to [here](./docs/pab.md).
