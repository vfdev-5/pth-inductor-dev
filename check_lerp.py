import torch

def func(x, y, z):
    return torch.lerp(x, y, z)


cfunc = torch.compile(func)

x, y, z = torch.rand(5), torch.rand(5), torch.rand(5)
out = cfunc(x, y, z)

print(out.shape)
