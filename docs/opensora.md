# Open-Sora

We support text-to-video generation for Open-Sora. Open-Sora is an open-source initiative dedicated to efficiently reproducing OpenAI's Sora which uses `DiT` model with `Spatial-Temporal Attention`.


## Inference

You can perform video inference for Open-Sora(v1.2) as follows.

```shell
# Use script
bash scripts/opensora/sample.sh
# Use command line
torchrun --standalone --nproc_per_node=1 scripts/opensora/sample.py --config configs/opensora/sample.yaml
```

We disable some speedup methods by default. You can change settings in yaml config. Here are details of some key arguments:

- `--nproc_per_node`: The GPU number you want to use for the current node. Multiple GPUs inference will enable Dynamic Sequence Parallel.
- `--dtype`: The data type for sampling. The default value is `bf16`.
- `--enable_flashattn`: Whether enable the FlashAttention. The default value is `False`. Recommend to enable.
- `--text_speedup`: Whether enable the T5 encoder optimization. This speeds up the text encoder. The default value is `False`. Requires apex install.

### Inference with [PAB](./docs/pab.md)

PAB provides more efficient inference at the cost of minor quality loss. You can run as follows:

```shell
# Use script
bash scripts/opensora/sample_pab.sh
# Use command line
torchrun --standalone --nproc_per_node=8 scripts/opensora/sample.py --config configs/opensora/sample_pab.yaml
```

You can change settings in yaml config. Here are details of some key arguments for training:

- `--nproc_per_node`: The GPU number you want to use for the current node. Multiple GPUs inference will enable Dynamic Sequence Parallel.

For the more detailed args and usages related with PAB, please refer to [here](./docs/pab.md).

### Training

Training is under development now. Will be available soon.
