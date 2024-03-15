import torch
from org_dit import DiT_S_2 as ORG_MODEL
from torch import nn

from opendit.models.dit.dit import DiT_S_2 as NEW_MODEL


def initialize_weights(model):
    # use normal distribution to initialize all weights for test

    def _basic_init(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.normal_(module.bias, std=0.02)

    model.apply(_basic_init)

    # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
    w = model.x_embedder.proj.weight.data
    nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
    nn.init.normal_(model.x_embedder.proj.bias, std=0.02)

    # Initialize label embedding table:
    nn.init.normal_(model.y_embedder.embedding_table.weight, std=0.02)

    # Initialize timestep embedding MLP:
    nn.init.normal_(model.t_embedder.mlp[0].weight, std=0.02)
    nn.init.normal_(model.t_embedder.mlp[2].weight, std=0.02)

    # Zero-out adaLN modulation layers in DiT blocks:
    for block in model.blocks:
        nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.02)
        nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.02)

    # Zero-out output layers:
    nn.init.normal_(model.final_layer.adaLN_modulation[-1].weight, std=0.02)
    nn.init.normal_(model.final_layer.adaLN_modulation[-1].bias, std=0.02)
    nn.init.normal_(model.final_layer.linear.weight, std=0.02)
    nn.init.normal_(model.final_layer.linear.bias, std=0.02)


def test_model():
    torch.manual_seed(0)
    org_model = ORG_MODEL().cuda()
    initialize_weights(org_model)
    torch.manual_seed(0)
    new_model = NEW_MODEL().cuda()
    initialize_weights(new_model)

    # Check if the model parameters are equal
    for (org_name, org_param), (new_name, new_param) in zip(org_model.named_parameters(), new_model.named_parameters()):
        assert org_name == new_name
        assert torch.equal(org_param, new_param), f"Parameter {org_name} is not equal\n{org_param}\n {new_param}"

    x1 = torch.randn(2, 4, 32, 32).cuda().requires_grad_(True)
    y1 = torch.randint(0, 10, (2,)).cuda()
    t1 = torch.randint(0, 10, (2,)).cuda()
    x2 = x1.clone().detach().requires_grad_(True)
    y2 = y1.clone().detach()
    t2 = t1.clone().detach()

    org_output = org_model(x1, t1, y1)
    new_output = new_model(x2, t2, y2)
    assert torch.allclose(
        org_output, new_output, atol=1e-5
    ), f"Max diff: {torch.max(torch.abs(org_output - new_output))}, Mean diff: {torch.mean(torch.abs(org_output - new_output))}"

    org_output.mean().backward()
    new_output.mean().backward()
    assert torch.allclose(
        x1.grad, x2.grad, atol=1e-5
    ), f"Max diff: {torch.max(torch.abs(x1.grad - x2.grad))}, Mean diff: {torch.mean(torch.abs(x1.grad - x2.grad))}"


if __name__ == "__main__":
    test_model()
