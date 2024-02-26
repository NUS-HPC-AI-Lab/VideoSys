<p align="center">
<img width="200px" alt="OpenMoE" src="./figure/logo.png?raw=true">
</p>
<p align="center">| <a href="https://github.com/oahzxl/OpenDiT">[Homepage]</a> |</p>
</p>

# About

OpenDiT is an open-source project that provides a high-performance implementation of Diffusion Transformer(DiT) powered by Colossal-AI. Specifically designed to enhance the efficiency of training and inference for DiT applications involving text-to-video and text-to-image generation.

OpenDiT boasts the following characteristics:

1. Up to 2x-3x speedup and 50% memory reduction on GPU
      * Kernel optimization including FlashAttention, Fused AdaLN, and Fused layernorm kernel.
      * Hybrid parallelism methods including ZeRO, Gemini, and DDP. And shard ema model to further reduce memory cost.
2. FastSeq: A novel sequence parallelism method
    * Sepcially designed for DiT-like workload where activation is large but parameter is small.
    * Up to 48% communication save for intra-node sequence parallel.
    * Break the memory limit of single GPU and reduce the overall training and inference time.
3. Ease of use
    * Huge performance gains with a few lines changes
    * You don't need to care about how the parallel part is implemented
4. Complete pipeline of text-to-image and text-to-video generation
    * User can easily use and adapt our pipeline to their own research without modifying the parallel part.
    * Verify the accuracy of OpenDiT with text-to-image training on ImageNet and release checkpoint.

Authors: [Xuanlei Zhao](https://oahzxl.github.io/), [Zhongkai Zhao](https://www.linkedin.com/in/zhongkai-zhao-kk2000/), [Ziming Liu](https://maruyamaaya.github.io/), [Haotian Zhou](https://github.com/ht-zhou), [Qianli Ma](https://fazzie-key.cool/about/index.html), [Yang You](https://www.comp.nus.edu.sg/~youy/).

## Installation

Prerequisites:

-   Python >= 3.10
-   PyTorch >= 1.13 (We recommend to use a >2.0 version)
-   CUDA >= 11.6

We strongly recommend you use Anaconda to create a new environment (Python >= 3.10) to run our examples:

```shell
conda create -n opendit python=3.10 -y
conda activate opendit
```

Install [ColossalAI](https://github.com/hpcaitech/ColossalAI):

```shell
git clone https://github.com/hpcaitech/ColossalAI.git
cd PATH/TO/ColossalAI
git checkout adae123df3badfb15d044bd416f0cf29f250bc86
pip install -e .
```

Install OpenDiT:

```shell
git clone https://github.com/oahzxl/OpenDiT
cd PATH/TO/OpenDiT
pip install -e .
```

Install other required libraries:

```shell
pip install -r requirements.txt
```

(Optional but recommend) Install libraries for training & inference speed up:

```shell
# Install Triton for fused adaln kernel
pip install triton

# Install FlashAttention
pip install flash-attn

# Install apex for fused layernorm kernel
git clone https://github.com/NVIDIA/apex.git
cd PATH/TO/apex
git checkout 741bdf50825a97664db08574981962d66436d16a
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./ --global-option="--cuda_ext" --global-option="--cpp_ext"
```


## Usage

You can train the DiT model by executing the following command:

```shell
# Train the DiT model
bash train_img.sh
```

Here are details of some key arguments for training:
- `--plugin`: The booster plugin used by ColossalAI, `zero2` and `ddp` are supported.
- `--mixed_precision`: The data type for mixed precision training. The default value is `bf16`.
- `--grad_checkpoint`: Whether enable the gradient checkpointing. This saves the memory cost during training process. The default value is `False`.
- `--enable_modulate_kernel`: Whether enable the modulate kernel optimization. This speeds up the training process. The default value is `False`.
- `--enable_layernorm_kernel`: Whether enable the layernorm kernel optimization. This speeds up the training process. The default value is `False`.
- `--enable_flashattn`: Whether enable the FlashAttention. This speeds up the training process. The default value is `False`.
- `--sequence_parallel_size`: The sequence parallelism size. Will enable sequence parallelism when setting a value > 1. The defualt value is 1.
- `--sequence_parallel_type`: The sequence parallelism type you choose. The defualt value is `None`.

For more details of the configuration of the training process, please visit our code.

You can perform inference using DiT model by executing the following command:

```shell
# Inference using the trained DiT model
bash sample.sh
```

## Core Design
### Efficient Sequence Parallelism -- （AllGanther）

### Overlapped QKV

### Distributed Attention

### Communication Complexity Analysis

## DiT Reproduction Result

![Image text](./figure/loss.png)

## Acknowledgement

## Contributing

If you encounter problems using OpenDiT, feel free to create an issue! We also welcome pull requests from the community.

## Citation
```
@misc{zhao2024opendit,
  author = {Xuanlei Zhao, Zhongkai Zhao, Ziming Liu, Haotian Zhou, Qianli Ma, and Yang You},
  title = {OpenDiT},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/oahzxl/OpenDiT}},
}
```
