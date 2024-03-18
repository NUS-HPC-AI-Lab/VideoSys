# DiT

We support label-to-image generation for DiT.

### Training

You can train the DiT model on CIFAR10 by executing the following command:

```shell
# Use script
bash scripts/dit/train_dit.sh
# Use command line
torchrun --standalone --nproc_per_node=2 scripts/dit/train.py \
    --model DiT-XL/2 \
    --batch_size 2 \
    --num_classes 10
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
- `--num_classes`: Label class number. Should be 10 for CIFAR10 and 1000 for ImageNet. Only used for label-to-image generation.

For more details on the configuration of the training process, please visit our code.

### Multi-Node Training

To train OpenDiT on multiple nodes, you can use the following command:

```
colossalai run --nproc_per_node 8 --hostfile hostfile train.py \
    --model DiT-XL/2 \
    --batch_size 2 \
    --num_classes 10
```

And you need to create `hostfile` under the current dir. It should contain all IP address of your nodes and you need to make sure all nodes can be connected without password by ssh. An example of hostfile:

```
111.111.111.111 # ip of node1
222.222.222.222 # ip of node2
```

Or you can use the standard `torchrun` to launch multi-node training as well.

### Inference

You can perform inference using DiT model as follows. You need to replace the checkpoint path to your own trained model. Or you can download [official](https://github.com/facebookresearch/DiT?tab=readme-ov-file#sampling--) or [our](https://drive.google.com/file/d/1P4t2V3RDNcoCiEkbVWAjNetm3KC_4ueI/view?usp=drive_link) checkpoint for inference.

```shell
# Use script
bash scripts/dit/sample_dit.sh
# Use command line
python scripts/dit/sample_dit.py \
    --model DiT-XL/2 \
    --image_size 256 \
    --num_classes 10 \
    --ckpt ckpt_path
```

Here are details of some addtional key arguments for inference:

- `--ckpt`: The weight of ema model `ema.pt`. To check your training progress, it can also be our saved base model `epochXX-global_stepXX/model`, it will produce better results than ema in early training stage.
- `--num_classes`: Label class number. Should be 10 for CIFAR10, and 1000 for ImageNet (including official and our checkpoint).


### Reproduction

To reproduce our results, you need to change the dataset in `scripts/dit/train_dit.py` and execute the following command:

```
torchrun --standalone --nproc_per_node=8 scripts/dit/train_dit.py \
    --model DiT-XL/2 \
    --batch_size 180 \
    --enable_layernorm_kernel \
    --enable_flashattn \
    --mixed_precision bf16 \
    --num_classes 1000
```
