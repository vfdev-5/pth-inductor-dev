import torch


@torch.compile()
def fn(x):
    return 0 > x


x = torch.randn(512, device="cuda")
fn(x)