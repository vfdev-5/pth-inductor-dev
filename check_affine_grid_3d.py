# TORCH_COMPILE_DEBUG=1 python check_affine_grid_3d.py

import os

import torch

from torch.nn.functional import affine_grid


if not ("OMP_NUM_THREADS" in os.environ):
    torch.set_num_threads(4)


print("")
print(f"Torch version: {torch.__version__}")
print(f"Torch config: {torch.__config__.show()}")
print("")

torch.set_printoptions(precision=7)


def transform(img, theta, align_corners):
    n, c, d, h, w = img.shape
    grid = affine_grid(theta, size=(n, c, d, h, w), align_corners=align_corners)
    return grid

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
mode = "bilinear"

c_transform = torch.compile(transform)


for n in [8, ]:

    a = torch.deg2rad(torch.tensor(45.0))
    ca, sa = torch.cos(a), torch.sin(a)
    s1 = 1.23
    s2 = 1.34
    s3 = 1.45

    c, d, h, w = 3, 13, 11, 12

    theta = torch.tensor([[
        [ca / s1, sa,   0.123 * sa,   0.1],
        [-sa, ca / s2,  sa,   0.2],
        [-0.123 * sa, -sa, ca / s3,   0.3],
    ]])
    theta = theta.expand(n, 3, 4).contiguous()
    theta = theta.to(device=device, dtype=dtype)

    x = torch.arange(n * c * d * h * w, device=device).reshape(n, c, d, h, w).to(torch.uint8)
    x = x.to(dtype=dtype)
    x = x.contiguous(memory_format=memory_format)

    expected = transform(x, theta, align_corners)
    output = c_transform(x, theta, align_corners)

    print("input:", x.shape, x.stride(), x.dtype)
    print("output:", output.shape, output.stride(), output.dtype)
    print("expected:", expected.shape, expected.stride(), expected.dtype)

    # assert x.stride() == output.stride(), (x.stride(), output.stride())

    # adiff = (output.float() - expected.float()).abs()
    # m = adiff > 1e-3
    # print("adiff:", adiff[m][:7])
    # print("output vs expected:", [
    #     (a.item(), b.item()) for a, b in zip(output[m][:7], expected[m][:7])
    # ])
    torch.testing.assert_close(output, expected)
