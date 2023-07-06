# TORCH_COMPILE_DEBUG=1 python check_interpolate.py

import os
from typing import Optional, Tuple, Union, List

import torch
import torchvision

torchvision.disable_beta_transforms_warning()

if not ("OMP_NUM_THREADS" in os.environ):
    torch.set_num_threads(1)


from torch import nn, Tensor


def transform(img):
    img = img[None, ...]
    # img = torch.nn.functional.interpolate(img, size=(270, 270), mode="bilinear", antialias=False)

    img = torch.nn.functional.interpolate(img, size=(12, 12), mode="bicubic", antialias=False)

    return img

c_transform = torch.compile(transform)

# memory_format = torch.channels_last
memory_format = torch.contiguous_format


# x = torch.randint(0, 256, size=(1, 3, 345, 456), dtype=torch.uint8)
# x = torch.arange(3 * 345 * 456).reshape(1, 3, 345, 456).to(torch.uint8)
x = torch.arange(3 * 32 * 32).reshape(1, 3, 32, 32).to(torch.uint8).to(torch.float32)
x = x.contiguous(memory_format=memory_format)[0]

output = c_transform(x)
expected = transform(x)

print(output.dtype, expected.dtype)
print(output.shape, expected.shape)
print(output.stride(), expected.stride())

print(output[0, 0, :3, :5])
print(expected[0, 0, :3, :5])

torch.testing.assert_close(output, expected)
