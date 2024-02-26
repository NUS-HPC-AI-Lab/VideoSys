# OpenDiT

## Introduction
OpenDiT is an open-source project that focuses on exploring Diffusion Models with Transformers (DiT). Its goal is to offer the open-source community a machine learning framework that enables efficient training and inference of DiT models, facilitating research related to DiT.

In this repository, we have developed an efficient DiT training and inference framework utilizing PyTorch and ColossalAI, achieving a substantial enhancement compared to the existing open-source alternatives. We have implemented both the image DiT and video DiT paradigms to accommodate diverse application needs.

Concerning DiT, we have made some enhancements to sequence parallelism, aiming to reduce communication consumption and boost throughput during the training and inference of DiTs. Compared with the native pytorch solution, we achieved a 3x end-to-end speed improvement; compared with the current SOTA solution Ulysses, our communication volume was reduced by up to 50%.

## Set Up
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
### Install kernels to speed up
```
# triton for modulate kernel
pip install triton

# flash attention
pip install flash-attn

# apex layernorm
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

## Experimental Result

## Acknowledgement

## License


