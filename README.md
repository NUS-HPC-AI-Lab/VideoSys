# OpenDiT

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

Core contributors: [Xuanlei Zhao](https://oahzxl.github.io/), [Zhongkai Zhao](https://www.linkedin.com/in/zhongkai-zhao-kk2000/), [Ziming Liu](https://maruyamaaya.github.io/), [Haotian Zhou](https://github.com/ht-zhou), [Qianli Ma](https://fazzie-key.cool/about/index.html).

## Installation
### Install ColossalAI
```
git clone https://github.com/hpcaitech/ColossalAI.git
cd ColossalAI
git checkout adae123df3badfb15d044bd416f0cf29f250bc86
pip install -e .
```
### Install OpenDiT
```
# Prerequisite
cd OpenDiT
pip install -e .
```
### (Optional) Install kernels to speed up
```
# triton for fused adaln kernel
pip install triton

# flash attention
pip install flash-attn

# apex for fused layernorm kernel
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 741bdf50825a97664db08574981962d66436d16a
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./ --global-option="--cuda_ext" --global-option="--cpp_ext"
```

## Usage
### Image Pipeline
```
# train
bash train_img.sh
# inference
bash sample.sh
```
### Video Pipeline
```
# train
bash preprocess.sh
bash train_video.sh
```

## Core Design
### Efficient Sequence Parallelism -- （AllGanther）

### Overlapped QKV

### Distributed Attention

### Communication Complexity Analysis

## DiT Reproduction Result

## Acknowledgement

## Citation
