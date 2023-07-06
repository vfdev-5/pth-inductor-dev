# TORCH_COMPILE_DEBUG=1 python check_affine_grid_sample_graph.py

import os

import torch
from torch.nn.functional import grid_sample, affine_grid


if not ("OMP_NUM_THREADS" in os.environ):
    torch.set_num_threads(4)


def transform(img, theta):
    n, c, h, w = img.shape
    grid = affine_grid(theta, size=(1, c, h, w), align_corners=False)
    grid = grid.expand(n, h, w, 2).to(device=img.device, dtype=img.dtype)
    output = grid_sample(img, grid, align_corners=False)
    return output

# def transform(img, theta):
#     n, c, h, w = img.shape
#     grid = affine_grid(theta, size=(n, c, h, w))
#     return grid


device = "cuda"


# x = torch.randint(0, 256, size=(1, 3, 270, 456), dtype=torch.uint8)
# x = torch.randint(0, 256, size=(1, 3, 345, 270), dtype=torch.uint8)

# x = torch.randint(0, 256, size=(1, 3, 345, 456), dtype=torch.uint8)
# x = torch.arange(3 * 345 * 456).reshape(1, 3, 345, 456).to(torch.uint8)
# x = torch.randint(0, 256, size=(1, 3, 400, 400), dtype=torch.uint8)
x = torch.arange(3 * 345 * 456, device=device).reshape(1, 3, 345, 456).to(torch.uint8).to(torch.float32)
# x = torch.arange(3 * 32 * 32, device=device).reshape(1, 3, 32, 32).to(torch.uint8).to(torch.float32)
# x = x.contiguous(memory_format=memory_format)[0]


a = torch.deg2rad(torch.tensor(45.0))
ca, sa = torch.cos(a), torch.sin(a)
s1 = 1.23
s2 = 1.34
theta = torch.tensor([[
    [ca / s1, sa, 0.0],
    [-sa, ca / s2, 0.0],
]], device=device, dtype=x.dtype)

expected = transform(x, theta)

c_transform = torch.compile(transform)
output = c_transform(x, theta)

print(output.dtype, expected.dtype)
print(output.shape, expected.shape)
print(output.stride(), expected.stride())

torch.testing.assert_close(output, expected)

