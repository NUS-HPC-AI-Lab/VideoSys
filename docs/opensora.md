# OpenSora

https://github.com/hpcaitech/Open-Sora

We support text-to-video generation for OpenSora.

### Training

We current support `VDiT` and `Latte` for video generation. VDiT adopts DiT structure and use video as inputs data. Latte further use more efficient spatial & temporal blocks based on VDiT (not exactly align with origin [Latte](https://github.com/Vchitect/Latte)).

Our video training pipeline is a faithful implementation, and we encourage you to explore your own strategies using OpenDiT. You can train the video DiT model by executing the following command:

```shell
# train with scipt
bash train_video.sh
# train with command line
# model can also be Latte-XL/1x2x2
torchrun --standalone --nproc_per_node=2 train.py \
    --model VDiT-XL/1x2x2 \
    --use_video \
    --data_path ./videos/demo.csv \
    --batch_size 1 \
    --num_frames 16 \
    --image_size 256 \
    --frame_interval 3
```

We disable all speedup methods by default. Here are details of some key arguments for training:

- `--nproc_per_node`: The GPU number you want to use for the current node.
- `--plugin`: The booster plugin used by ColossalAI, `zero2` and `ddp` are supported. The default value is `zero2`. Recommend to enable `zero2`.
- `--mixed_precision`: The data type for mixed precision training. The default value is `bf16`.
- `--grad_checkpoint`: Whether enable the gradient checkpointing. This saves the memory cost during training process. The default value is `False`. Recommend to enable.
- `--enable_layernorm_kernel`: Whether enable the layernorm kernel optimization. This speeds up the training process. The default value is `False`. Recommend to enable it.
- `--enable_flashattn`: Whether enable the FlashAttention. This speeds up the training process. The default value is `False`. Recommend to enable.
- `--enable_modulate_kernel`: Whether enable the modulate kernel optimization. This speeds up the training process. The default value is `False`. This kernel will cause NaN under some circumstances. So we recommend to disable it for now.
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

You can perform video inference using DiT model as follows. We are still working on the video ckpt.

```shell
# Use script
bash sample_video.sh
# Use command line
# model can also be Latte-XL/1x2x2
python sample.py \
    --model VDiT-XL/1x2x2 \
    --use_video \
    --ckpt ckpt_path \
    --num_frames 16 \
    --image_size 256 \
    --frame_interval 3
```

Inference tips: 1) EMA model requires quite long time to converge and produce meaningful results. So you can sample base model (`--ckpt /epochXX-global_stepXX/model`) instead of ema model (`--ckpt /epochXX-global_stepXX/ema.pt`) to check your training process. But ema model should be your final result. 2) Modify the text condition in `sample.py` which aligns with your datasets helps to produce better results in the early stage of training.

### Data Preparation

You can follow the [instruction](https://github.com/hpcaitech/Open-Sora/tree/release?tab=readme-ov-file#data-processing) to prepare the data for training.
