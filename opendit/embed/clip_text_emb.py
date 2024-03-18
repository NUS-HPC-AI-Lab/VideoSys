import numpy
import torch
import torch.nn as nn
import transformers
from timm.models.vision_transformer import Mlp
from transformers import CLIPTextModel, CLIPTokenizer

transformers.logging.set_verbosity_error()


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""

    def __init__(self, path="openai/clip-vit-huge-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(path)
        self.transformer = CLIPTextModel.from_pretrained(path)
        self.device = device
        self.max_length = max_length
        self._freeze()

    def _freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        pooled_z = outputs.pooler_output
        return z, pooled_z

    def encode(self, text):
        return self(text)


class TextEmbedder(nn.Module):
    """
    Embeds text prompt into vector representations. Also handles text dropout for classifier-free guidance.
    """

    def __init__(self, path, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.text_encoder = FrozenCLIPEmbedder(path=path)
        self.dropout_prob = dropout_prob

        output_dim = self.text_encoder.transformer.config.hidden_size
        self.output_projection = nn.Linear(output_dim, hidden_size)

    def token_drop(self, text_prompts, force_drop_ids=None):
        """
        Drops text to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = numpy.random.uniform(0, 1, len(text_prompts)) < self.dropout_prob
        else:
            # TODO
            drop_ids = force_drop_ids == 1
        labels = list(numpy.where(drop_ids, "", text_prompts))
        # print(labels)
        return labels

    def forward(self, text_prompts, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            text_prompts = self.token_drop(text_prompts, force_drop_ids)
        embeddings, pooled_embeddings = self.text_encoder(text_prompts)
        # return embeddings, pooled_embeddings
        text_embeddings = self.output_projection(pooled_embeddings)
        return text_embeddings


class CaptionEmbedder(nn.Module):
    """
    copied from https://github.com/hpcaitech/Open-Sora

    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    def __init__(self, in_channels, hidden_size, uncond_prob, act_layer=nn.GELU(approximate="tanh"), token_num=120):
        super().__init__()

        self.y_proj = Mlp(
            in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size, act_layer=act_layer, drop=0
        )
        self.register_buffer("y_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels**0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0]).cuda() < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None, None], self.y_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        if train:
            assert caption.shape[2:] == self.y_embedding.shape
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        caption = self.y_proj(caption)
        return caption
