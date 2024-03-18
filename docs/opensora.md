# OpenSora

https://github.com/hpcaitech/Open-Sora

We support text-to-video generation for OpenSora. Open-Sora is an open-source initiative dedicated to efficiently reproducing OpenAI's Sora which uses `DiT` model with `Spatial-Temporal Attention`.


### Training

You can train the video DiT model by executing the following command:

```shell
# train with scipt
bash scripts/opensora/train_opensora.sh
# train with command line
torchrun --standalone --nproc_per_node=2 scripts/opensora/train_opensora.py \
    --batch_size 2 \
    --mixed_precision bf16 \
    --lr $LR \
    --grad_checkpoint \
    --data_path "./videos/demo.csv" \
    --model_pretrained_path "ckpt_path"
```

We disable all speedup methods by default. Here are details of some key arguments for training:

- `--nproc_per_node`: The GPU number you want to use for the current node.
- `--plugin`: The booster plugin used by ColossalAI, `zero2` and `ddp` are supported. The default value is `zero2`. Recommend to enable `zero2`.
- `--mixed_precision`: The data type for mixed precision training. The default value is `bf16`.
- `--grad_checkpoint`: Whether enable the gradient checkpointing. This saves the memory cost during training process. The default value is `False`. Recommend to enable.
- `--enable_layernorm_kernel`: Whether enable the layernorm kernel optimization. This speeds up the training process. The default value is `False`. Recommend to enable it.
- `--enable_flashattn`: Whether enable the FlashAttention. This speeds up the training process. The default value is `False`. Recommend to enable.
- `--enable_modulate_kernel`: Whether enable the modulate kernel optimization. This speeds up the training process. The default value is `False`. This kernel will cause NaN under some circumstances. So we recommend to disable it for now.
- `--text_speedup`: Whether enable the T5 encoder optimization. This speeds up the training process. The default value is `False`. Requires apex install.
- `--load`: Load previous saved checkpoint dir and continue training.


### Multi-Node Training

To train OpenDiT on multiple nodes, you can use the following command:

```
colossalai run --nproc_per_node 8 --hostfile hostfile xx.py \
    --XXX
```

And you need to create `hostfile` under the current dir. It should contain all IP address of your nodes and you need to make sure all nodes can be connected without password by ssh. An example of hostfile:

```
111.111.111.111 # ip of node1
222.222.222.222 # ip of node2
```

Or you can use the standard `torchrun` to launch multi-node training as well.


### Inference

You can perform video inference using DiT model as follows. Model weights can be downloaded [here](https://github.com/hpcaitech/Open-Sora/tree/main?tab=readme-ov-file#model-weights).

```shell
# Use script
bash scripts/opensora/sample_opensora.sh
# Use command line
python scripts/opensora/sample_opensora.py \
    --model_time_scale 1 \
    --model_space_scale 1 \
    --image_size 512 512 \
    --num_frames 16 \
    --fps 8 \
    --dtype fp16 \
    --model_pretrained_path "ckpt_path"
```

Inference tips: 1) EMA model requires quite long time to converge and produce meaningful results. So you can sample base model (`--ckpt /epochXX-global_stepXX`) instead of ema model (`--ckpt /epochXX-global_stepXX/ema.pt`) to check your training process. But ema model should be your final result. 2) Modify the text condition in `sample.py` which aligns with your datasets helps to produce better results in the early stage of training.

### Data Preparation

You can follow the [instruction](https://github.com/hpcaitech/Open-Sora/tree/release?tab=readme-ov-file#data-processing) to prepare the data for training.
