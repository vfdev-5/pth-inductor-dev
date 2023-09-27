# TORCH_COMPILE_DEBUG=1 python check_affine_grid_sampler_fused.py

import os

import torch

# from torch.nn.functional import grid_sample, affine_grid
from torch._decomp.decompositions import _affine_grid_generator_4d as _affine_grid, _grid_sampler_2d as _grid_sample


if not ("OMP_NUM_THREADS" in os.environ):
    torch.set_num_threads(4)


print("")
print(f"Torch version: {torch.__version__}")
print(f"Torch config: {torch.__config__.show()}")
print("")

torch.set_printoptions(precision=7)


def transform(img, theta, align_corners, mode):
    n, c, h, w = img.shape
    grid = _affine_grid(theta, size=(n, c, h, w), align_corners=align_corners)
    output = _grid_sample(img, grid, align_corners=align_corners, interpolation_mode=mode)
    return output


a = torch.deg2rad(torch.tensor(45.0))
s1 = 1.23
s2 = 1.34
ca, sa = torch.cos(a), torch.sin(a)

# device = "cpu"
device = "cuda"

torch.manual_seed(12)
num_threads = 1
torch.set_num_threads(num_threads)

memory_format = torch.contiguous_format
# memory_format = torch.channels_last
# dtype = torch.float64
dtype = torch.float32

align_corners = False
# mode = "nearest"
# mode = "bicubic"
# mode = "bilinear"
mode = 0

c_transform = torch.compile(transform)


for n in [8, ]:

    c, h, w = 3, 345, 456
    theta = torch.tensor([[
        [ca / s1, sa, 0.0],
        [-sa, ca / s2, 0.0],
    ]])
    theta = theta.expand(n, 2, 3).contiguous()
    x = torch.arange(n * c * h * w, device=device).reshape(n, c, h, w).to(torch.uint8)
    theta = theta.to(device=device, dtype=dtype)

    x = x.to(dtype=dtype)
    x = x.contiguous(memory_format=memory_format)

    output = c_transform(x, theta, align_corners, mode)
    # expected = transform(x, theta, align_corners, mode)

    print("input:", x.shape, x.stride(), x.dtype)
    print("output:", output.shape, output.stride(), output.dtype)
    # print("expected:", expected.shape, expected.stride(), expected.dtype)

    # assert x.stride() == output.stride(), (x.stride(), output.stride())

    # adiff = (output.float() - expected.float()).abs()
    # m = adiff > 1e-3
    # print("adiff:", adiff[m][:7])
    # print("output vs expected:", [
    #     (a.item(), b.item()) for a, b in zip(output[m][:7], expected[m][:7])
    # ])
    # torch.testing.assert_close(output, expected)
