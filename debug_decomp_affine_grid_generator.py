
import os

from typing import List
import torch
from torch import Tensor
from torch.nn.functional import affine_grid


if not ("OMP_NUM_THREADS" in os.environ):
    torch.set_num_threads(4)


def _linspace_from_neg_one(
    num_steps: int, align_corners: bool, dtype: torch.dtype, device: torch.device
):
    if num_steps <= 1:
        return torch.tensor(0, device=device, dtype=dtype)

    a = ((num_steps - 1) / num_steps) if not align_corners else 1
    return torch.linspace(-a, a, steps=num_steps, device=device, dtype=dtype)


def _make_base_grid_4d(theta: Tensor, h: int, w: int, align_corners: bool):
    dtype = theta.dtype
    device = theta.device

    # Using padding and summation generates a single kernel vs using torch.stack where 3 kernels generated
    # corresponding to each individual tensor: grid_x, grid_y, grid_one
    grid_x = (
        _linspace_from_neg_one(w, align_corners, dtype, device)
        .view(1, w, 1)
        .expand(h, w, 1)
    )
    grid_y = (
        _linspace_from_neg_one(h, align_corners, dtype, device)
        .view(h, 1, 1)
        .expand(h, w, 1)
    )
    grid_one = torch.ones((1, 1, 1), dtype=dtype, device=device)

    # this is just a temporary hack and we should use torch.stack here once #104480 is merged
    grid_x = torch.nn.functional.pad(grid_x, pad=(0, 2), mode="constant", value=0)
    grid_y = torch.nn.functional.pad(grid_y, pad=(1, 1), mode="constant", value=0)
    grid_one = torch.nn.functional.pad(grid_one, pad=(2, 0), mode="constant", value=0)
    return grid_x + grid_y + grid_one


def decomp_affine_grid(img: Tensor, theta: Tensor, align_corners: bool):
    n, _, h, w = img.shape
    base_grid = _make_base_grid_4d(theta, h, w, align_corners=align_corners)
    # grid = (base_grid.flatten(0, 1).unsqueeze(-1) * theta.mT.unsqueeze(1)).sum(-2)
    grid = (base_grid.view(-1, 3, 1) * theta.mT.unsqueeze(1)).sum(-2)
    return grid.view(n, h, w, 2)


def eager_affine_grid(img, theta, align_corners: bool):
    n, c, h, w = img.shape
    grid = affine_grid(theta, size=(n, c, h, w), align_corners=align_corners)
    return grid


device = "cuda"
x = torch.arange(3 * 345 * 456, device=device).reshape(1, 3, 345, 456).to(torch.uint8).to(torch.float32)

a = torch.deg2rad(torch.tensor(45.0))
ca, sa = torch.cos(a), torch.sin(a)
s1 = 1.23
s2 = 1.34
# theta = torch.tensor([[
#     [ca / s1, sa, 0.0],
#     [-sa, ca / s2, 0.0],
# ]], device=device, dtype=x.dtype)

theta = torch.tensor([[
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
]], device=device, dtype=x.dtype)

align_corners = False
expected = eager_affine_grid(x, theta, align_corners)
output = decomp_affine_grid(x, theta, align_corners)

print("Decomp/Compiled:")
print(output[0, 0, :4, :7])
print("Eager:")
print(expected[0, 0, :4, :7])

if output.is_floating_point():
    m = (output - expected).abs() > 1e-5
    print("Decomp/Compiled:", output[m])
    print("Eager:", expected[m])

torch.testing.assert_close(output, expected)

