# TORCH_COMPILE_DEBUG=1 python -u check_interpolate_bicubic.py

import os

import torch
from torch import Tensor


def _upsample_cubic_convolution1(x: Tensor, A: float) -> Tensor:
    return ((A + 2) * x - (A + 3)) * x * x + 1


def _upsample_cubic_convolution2(x: Tensor, A: float) -> Tensor:
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A



A = -0.75
t = torch.rand(100)

# Test 1
c1 = _upsample_cubic_convolution1(t, A)
c2 = _upsample_cubic_convolution1(1.0 - t, A)

tt = torch.stack([t, 1.0 - t], dim=0)
c3 = _upsample_cubic_convolution1(tt, A)

c31, c32 = torch.unbind(c3, dim=0)

torch.testing.assert_close(c1, c31)
torch.testing.assert_close(c2, c32)


# Test 2
c1 = _upsample_cubic_convolution2(t + 1.0, A)
c2 = _upsample_cubic_convolution2(2.0 - t, A)

tt = torch.stack([t + 1.0, 2.0 - t], dim=0)
c3 = _upsample_cubic_convolution2(tt, A)

c31, c32 = torch.unbind(c3, dim=0)

torch.testing.assert_close(c1, c31)
torch.testing.assert_close(c2, c32)
