# This code is copied from https://github.com/wilson1yan/VideoGPT
# Copyright (c) 2021 Wilson Yan. All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import os

import gdown
import torch

from .vqvae import VQVAE


def download(id, fname, root=os.path.expanduser("~/.cache/videogpt")):
    os.makedirs(root, exist_ok=True)
    destination = os.path.join(root, fname)

    if os.path.exists(destination):
        return destination

    gdown.download(id=id, output=destination, quiet=False)
    return destination


_VQVAE = {
    "bair_stride4x2x2": "1iIAYJ2Qqrx5Q94s5eIXQYJgAydzvT_8L",  # trained on 16 frames of 64 x 64 images
    "ucf101_stride4x4x4": "1uuB_8WzHP_bbBmfuaIV7PK_Itl3DyHY5",  # trained on 16 frames of 128 x 128 images
    "kinetics_stride4x4x4": "1DOvOZnFAIQmux6hG7pN_HkyJZy3lXbCB",  # trained on 16 frames of 128 x 128 images
    "kinetics_stride2x4x4": "1jvtjjtrtE4cy6pl7DK_zWFEPY3RZt2pB",  # trained on 16 frames of 128 x 128 images
}


def load_vqvae(model_name, device=torch.device("cpu"), root=os.path.expanduser("~/.cache/videogpt")):
    assert model_name in _VQVAE, f"Invalid model_name: {model_name}"
    filepath = download(_VQVAE[model_name], model_name, root=root)
    vqvae = VQVAE.load_from_checkpoint(filepath).to(device)
    vqvae.eval()

    return vqvae
