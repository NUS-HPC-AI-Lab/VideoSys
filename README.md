# OpenDiT
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
## Image Pipeline
```
# train
bash train_img.sh
# inference
bash sample.sh
```
## Video Pipeline
```
# train
bash preprocess.sh
bash train_video.sh
```
## Install kernels to speed up
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
