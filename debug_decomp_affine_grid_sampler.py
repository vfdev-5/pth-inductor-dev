# TORCH_COMPILE_DEBUG=1 python debug_decomp_affine_grid_sampler

import os

import torch
from torch.nn.functional import grid_sample, affine_grid


if not ("OMP_NUM_THREADS" in os.environ):
    torch.set_num_threads(4)


def transform(img, theta):
    n, c, h, w = img.shape
    grid = affine_grid(theta, size=(n, c, h, w), align_corners=False)
    grid = grid.to(device=img.device, dtype=img.dtype)
    output = grid_sample(img, grid, align_corners=False)
    return output


# device = "cuda"
device = "cpu"
dtype = torch.float32
# memory_format = torch.contiguous_format
memory_format = torch.channels_last

n = 1
x = torch.arange(n * 3 * 345 * 456, device=device).reshape(n, 3, 345, 456).to(torch.uint8)
x = x.to(dtype=dtype)
x = x.contiguous(memory_format=memory_format)


a = torch.deg2rad(torch.tensor(45.0))
ca, sa = torch.cos(a), torch.sin(a)
s1 = 1.23
s2 = 1.34
theta = torch.tensor([[
    [ca / s1, sa, 0.0],
    [-sa, ca / s2, 0.0],
]], device=device, dtype=x.dtype)
theta = theta.expand(n, 2, 3).contiguous()

# theta = torch.tensor([[
#     [1.0, 0.0, 0.0],
#     [0.0, 1.0, 0.0],
# ]], device=device, dtype=x.dtype)

c_transform = torch.compile(transform)

expected = transform(x, theta)
output = c_transform(x, theta)

print(x.stride())
print(expected.stride())
print(output.stride())

exit(0)

torch.set_printoptions(precision=7)

print("Decomp/Compiled:")
print(output[0, 0, :4, :7])
print("Eager:")
print(expected[0, 0, :4, :7])
# print(x[0, 0, :4, :7])


if output.is_floating_point():
    adiff = (output - expected).abs()
    m = adiff > 1e-5
    print("Adiff:", adiff[m][:7])
    print("Decomp/Compiled:", output[m][:7])
    print("Eager:", expected[m][:7])
    # print("Eager:", x[m])

# torch.testing.assert_close(x, expected)
torch.testing.assert_close(output, expected)

