import torch
from org_dit import DiT_S_2 as ORG_MODEL

from opendit.models.dit import DiT_S_2 as NEW_MODEL


def test_model():
    torch.manual_seed(0)
    org_model = ORG_MODEL().cuda()
    torch.manual_seed(0)
    new_model = NEW_MODEL().cuda()

    # Check if the model parameters are equal
    for org_param, new_param in zip(org_model.parameters(), new_model.parameters()):
        assert torch.equal(org_param, new_param)

    x1 = torch.randn(2, 4, 32, 32).cuda().requires_grad_(True)
    y1 = torch.randint(0, 10, (2,)).cuda()
    t1 = torch.randint(0, 10, (2,)).cuda()
    x2 = x1.clone().detach().requires_grad_(True)
    y2 = y1.clone().detach()
    t2 = t1.clone().detach()

    org_output = org_model(x1, t1, y1)
    new_output = new_model(x2, t2, y2)
    assert torch.equal(org_output, new_output)

    org_output.mean().backward()
    new_output.mean().backward()
    assert torch.equal(x1.grad, x2.grad)


if __name__ == "__main__":
    test_model()
