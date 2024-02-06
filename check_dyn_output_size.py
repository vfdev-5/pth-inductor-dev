import torch


def fn(t: torch.Tensor, x: int) -> torch.Tensor:
    xx = [0.5 + 0.0 * x, x * 0.5]
    min_x = min(xx) + 0.0
    return t + min_x


cfn = torch.compile(fn, fullgraph=True, dynamic=True)

output = cfn(torch.tensor(0.0), 10)
print(output)
