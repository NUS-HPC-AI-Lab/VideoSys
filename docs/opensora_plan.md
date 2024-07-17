# Open-Sora-Plan

Open-Sora-Plan aims to reproduce Sora (Open AI T2V model), we wish the open source community contribute to this project.

## Inference

You can perform video inference for Open-Sora-Plan (v1.1) as follows.

```shell
# Use script
bash scripts/opensora_plan/sample.sh
# Use command line
ttorchrun --standalone --nproc_per_node=1 scripts/opensora_plan/sample.py --config configs/opensora_plan/sample_65f.yaml
```

You can change settings in yaml config. Here are details of some key arguments:

- `--nproc_per_node`: The GPU number you want to use for the current node. Multiple GPUs inference will enable Dynamic Sequence Parallel.

### Inference with [PAB](./docs/pab.md)

PAB provides more efficient inference at the cost of minor quality loss. You can run as follows:

```shell
# Use script
bash scripts/opensora_plan/sample_pab.sh
# Use command line
torchrun --standalone --nproc_per_node=8 scripts/opensora_plan/sample.py --config configs/opensora_plan/sample_65f_pab.yaml
```

You can change settings in yaml config. Here are details of some key arguments for training:

- `--nproc_per_node`: The GPU number you want to use for the current node. Multiple GPUs inference will enable Dynamic Sequence Parallel.

For the more detailed args and usages related with PAB, please refer to [here](./docs/pab.md).
