import os

import colossalai
import torch
from colossalai.booster import Booster
from colossalai.booster.plugin import LowLevelZeroPlugin
from torchvision.datasets.utils import download_url

pretrained_models = {
    "DiT-XL-2-512x512.pt": "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-512x512.pt",
    "DiT-XL-2-256x256.pt": "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt",
    "Latte-XL-2-256x256-ucf101.pt": "https://huggingface.co/maxin-cn/Latte/resolve/main/ucf101.pt",
    "PixArt-XL-2-256x256.pth": "PixArt-XL-2-256x256.pth",
    "PixArt-XL-2-SAM-256x256.pth": "PixArt-XL-2-SAM-256x256.pth",
    "PixArt-XL-2-512x512.pth": "PixArt-XL-2-512x512.pth",
    "PixArt-XL-2-1024-MS.pth": "PixArt-XL-2-1024-MS.pth",
}


def reparameter(ckpt, name=None):
    # copied from https://github.com/hpcaitech/Open-Sora

    if "DiT" in name:
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        del ckpt["pos_embed"]
    elif "Latte" in name:
        ckpt = ckpt["ema"]
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        del ckpt["pos_embed"]
        del ckpt["temp_embed"]
    elif "PixArt" in name:
        ckpt = ckpt["state_dict"]
        ckpt["x_embedder.proj.weight"] = ckpt["x_embedder.proj.weight"].unsqueeze(2)
        del ckpt["pos_embed"]
    return ckpt


def find_model(model_name):
    # copied from https://github.com/hpcaitech/Open-Sora
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    if model_name in pretrained_models:  # Find/download our pre-trained DiT checkpoints
        model = download_model(model_name)
        model = reparameter(model, model_name)
        return model
    else:  # Load a custom DiT checkpoint:
        assert os.path.isfile(model_name), f"Could not find DiT checkpoint at {model_name}"
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
        if "pos_embed_temporal" in checkpoint:
            del checkpoint["pos_embed_temporal"]
        if "pos_embed" in checkpoint:
            del checkpoint["pos_embed"]
        if "ema" in checkpoint:  # supports checkpoints from train.py
            checkpoint = checkpoint["ema"]
        return checkpoint


def download_model(model_name):
    # copied from https://github.com/hpcaitech/Open-Sora
    """
    Downloads a pre-trained DiT model from the web.
    """
    assert model_name in pretrained_models
    local_path = f"pretrained_models/{model_name}"
    if not os.path.isfile(local_path):
        os.makedirs("pretrained_models", exist_ok=True)
        web_path = pretrained_models[model_name]
        download_url(web_path, "pretrained_models", model_name)
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model


def load_from_sharded_state_dict(model, ckpt_path):
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29501"
    colossalai.launch_from_torch({})
    plugin = LowLevelZeroPlugin(
        stage=2,
        precision="fp32",
        initial_scale=2**16,
    )
    booster = Booster(plugin=plugin)
    model, _, _, _, _ = booster.boost(model=model)
    booster.load_model(model, os.path.join(ckpt_path, "model"))

    save_path = os.path.join(ckpt_path, "model_ckpt.pt")
    torch.save(model.module.state_dict(), save_path)
    print(f"Model checkpoint saved to {save_path}")


def load_checkpoint(model, ckpt_path, save_as_pt=True):
    # copied from https://github.com/hpcaitech/Open-Sora
    if ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
        state_dict = find_model(ckpt_path)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")
    elif os.path.isdir(ckpt_path):
        load_from_sharded_state_dict(model, ckpt_path)
        if save_as_pt:
            save_path = os.path.join(ckpt_path, "model_ckpt.pt")
            torch.save(model.state_dict(), save_path)
    else:
        raise ValueError(f"Invalid checkpoint path: {ckpt_path}")
