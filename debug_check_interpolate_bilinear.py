# TORCH_COMPILE_DEBUG=1 python debug_check_interpolate_bilinear.py

import os
from typing import Optional, Tuple, Union, List

try:
    import numpy as np
    import cv2
    has_opencv = True
except ImportError:
    has_opencv = False

import torch
import torchvision

torchvision.disable_beta_transforms_warning()

if not ("OMP_NUM_THREADS" in os.environ):
    torch.set_num_threads(1)


from torch import nn, Tensor


def transform(img):
    img = img[None, ...]
    img = torch.nn.functional.interpolate(img, size=(270, 270), mode="bilinear", antialias=False)
    return img

c_transform = torch.compile(transform)


import torch._prims_common as utils
from torch._decomp.decompositions import pw_cast_for_opmath
aten = torch._ops.ops.aten


@pw_cast_for_opmath
def upsample_bilinear2d(
    input: Tensor,
    output_size: List[int],
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> Tensor:
    # get dimensions of original image
    n_batch, n_channels, in_h, in_w = input.shape

    out_h = output_size[0]
    out_w = output_size[1]

    # Calculate horizontal and vertical scaling factor
    # TODO: Figure out if scales_h/scales_w matters here
    if out_h > 1:
        if align_corners:
            h_scale_factor = (in_h - 1) / (out_h - 1)
        else:
            h_scale_factor = 1.0 / scales_h if scales_h is not None else in_h / out_h
    else:
        h_scale_factor = 0.0

    if out_w > 1:
        if align_corners:
            w_scale_factor = (in_w - 1) / (out_w - 1)
        else:
            w_scale_factor = 1.0 / scales_w if scales_w is not None else in_w / out_w
    else:
        w_scale_factor = 0.0

    i = torch.arange(out_h, dtype=torch.int64, device=input.device)
    j = torch.arange(out_w, dtype=torch.int64, device=input.device)

    if align_corners:
        x = h_scale_factor * i
        y = w_scale_factor * j
    else:
        x = (h_scale_factor * (i + 0.5) - 0.5).clamp(min=0.0)
        y = (w_scale_factor * (j + 0.5) - 0.5).clamp(min=0.0)

    x_floor = x.to(torch.int64).clamp(max=in_h - 1)
    x_ceil = (x_floor + 1).clamp(max=in_h - 1)
    y_floor = y.to(torch.int64).clamp(max=in_w - 1)
    y_ceil = (y_floor + 1).clamp(max=in_w - 1)

    x_view = x.unsqueeze(1)
    x_floor_view = x_floor.unsqueeze(1)
    x_ceil_view = x_ceil.unsqueeze(1)

    v1 = aten._unsafe_index(input, [None, None, x_floor_view, y_floor])
    v2 = aten._unsafe_index(input, [None, None, x_floor_view, y_ceil])
    v3 = aten._unsafe_index(input, [None, None, x_ceil_view, y_floor])
    v4 = aten._unsafe_index(input, [None, None, x_ceil_view, y_ceil])

    xscale2 = (x_view - x_floor_view).clamp(0.0, 1.0)
    yscale2 = (y - y_floor).clamp(0.0, 1.0)

    q1 = v1 + torch.mul(v2 - v1, yscale2)
    q2 = v3 + torch.mul(v4 - v3, yscale2)
    result = q1 + torch.mul(q2 - q1, xscale2)

    # convert output to correct memory format, if necessary
    memory_format = utils.suggest_memory_format(input)

    # following "heuristic: only use channels_last path when it's faster than the contiguous path"
    if input.device.type == "cuda" and n_channels < 16:
        memory_format = torch.contiguous_format

    result = result.contiguous(memory_format=memory_format)

    if not input.is_floating_point():
        result = result.round()

    return result


def decomp_transform(img):
    img = img[None, ...]
    img = upsample_bilinear2d(img, (270, 270), align_corners=False)
    return img


torch.manual_seed(12)
# x = torch.randint(0, 256, size=(1, 3, 345, 456), dtype=torch.uint8)
# x = torch.arange(3 * 345 * 456).reshape(1, 3, 345, 456).to(torch.uint8)
x = torch.arange(3 * 345 * 456).reshape(1, 3, 345, 456).to(torch.uint8).to(torch.float32)
# x = torch.arange(3 * 345 * 456).reshape(1, 3, 345, 456).to(torch.uint8).to(torch.float64)
x = x.contiguous(memory_format=torch.channels_last)[0]
# x = x[0]

output = c_transform(x)
# output = decomp_transform(x)
expected = transform(x)

out_cv = None
if has_opencv:
    np_x = x.permute(1, 2, 0).contiguous().numpy()
    out_cv = cv2.resize(np_x, (270, 270), interpolation=cv2.INTER_LINEAR)


torch.set_printoptions(precision=6)

print(output.dtype, expected.dtype)
print(output.shape, expected.shape)
print(output.stride(), expected.stride())

print("Decomp/Compiled:")
print(output[0, 0, :4, :7])
print("Eager:")
print(expected[0, 0, :4, :7])

if out_cv is not None:
    print("Opencv:")
    print(out_cv[:4, :7, 0])

if output.is_floating_point():
    m = (output - expected).abs() > 1e-5
    print("Decomp/Compiled:", output[m])
    print("Eager:", expected[m])


torch.testing.assert_close(output, expected)
