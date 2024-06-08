import torch

from opendit.models.opensora.embed import TextEmbedder

if __name__ == "__main__":
    r"""
    Returns:

    Examples from CLIPTextModel:

    ```python
    >>> from transformers import AutoTokenizer, CLIPTextModel

    >>> model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    >>> tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    >>> inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> last_hidden_state = outputs.last_hidden_state
    >>> pooled_output = outputs.pooler_output  # pooled (EOS token) states
    ```"""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    text_encoder = TextEmbedder(path="openai/clip-vit-base-patch32", dropout_prob=0.00001).to(device)

    text_prompt = [
        ["a photo of a cat", "a photo of a cat"],
        ["a photo of a dog", "a photo of a cat"],
        ["a photo of a dog human", "a photo of a cat"],
    ]
    # text_prompt = ('None', 'None', 'None')
    output, pooled_output = text_encoder(text_prompts=text_prompt, train=False)
    # print(output)
    print(output.shape)
    print(pooled_output.shape)
    # print(output.shape)
