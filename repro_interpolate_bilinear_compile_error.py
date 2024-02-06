# TORCH_COMPILE_DEBUG=1 python repro_interpolate_bilinear_compile_error.py

import os

import torch
import torch.nn.functional as F


if not ("OMP_NUM_THREADS" in os.environ):
    torch.set_num_threads(1)


def transform(img, osize, align_corners):
    # img = F.interpolate(img, size=osize, mode="bilinear", antialias=False, align_corners=align_corners)
    img = torch.nn.functional.interpolate(img, size=osize, mode="bilinear", antialias=False, align_corners=align_corners)
    return img


device = "cpu"
align_corners = False

c_transform = torch.compile(transform, dynamic=True)

memory_format = torch.contiguous_format

torch.manual_seed(12)

x = torch.randint(0, 256, size=(1, 3, 2345, 2456), dtype=torch.float32, device=device)

x = x.contiguous(memory_format=memory_format)
x = x.to(device=device)

osize = (1234, 1345)

output = c_transform(x, osize, align_corners=align_corners)
expected = transform(x, osize, align_corners=align_corners)

kwargs = {}
if x.dtype == torch.uint8:
    kwargs = {
        "atol": 1.0,
        "rtol": 0.0,
    }

torch.testing.assert_close(output, expected, **kwargs)

