from typing import List, Optional
import torch
from torch import Tensor


def _compute_scale(in_size, out_size, align_corners, scale=None):
    if align_corners:
        return (in_size - 1.0) / (out_size - 1.0) if out_size > 1 else 0
    else:
        return 1.0 / scale if scale is not None and scale > 0 else in_size / out_size


def _compute_source_index(scale, dst_index, align_corners):
    if align_corners:
        return scale * dst_index
    else:
        return scale * (dst_index + 0.5) - 0.5

# # --- Single op for input load
# def upsample_bilinear2d(
#     input: Tensor,
#     output_size: List[int],
#     align_corners: bool,
#     scales_h: Optional[float] = None,
#     scales_w: Optional[float] = None,
# ) -> Tensor:
#     # get dimensions of original image
#     _, n_channels, in_h, in_w = input.shape

#     # Calculate horizontal and vertical scaling factor
#     h_scale_factor = _compute_scale(in_h, output_size[0], align_corners, scales_h)
#     w_scale_factor = _compute_scale(in_w, output_size[1], align_corners, scales_w)

#     print(h_scale_factor, w_scale_factor)

#     i = torch.arange(output_size[0], device=input.device)
#     j = torch.arange(output_size[1], device=input.device)

#     x_f32 = _compute_source_index(w_scale_factor, j, align_corners).clamp(min=0.0)
#     y_f32 = _compute_source_index(h_scale_factor, i, align_corners).clamp(min=0.0)
#     x = x_f32.to(torch.long)
#     y = y_f32.to(torch.long)

#     kx = torch.arange(2, device=input.device)
#     ky = torch.arange(2, device=input.device)
#     x_indices = torch.clamp(x.unsqueeze(dim=-1) + kx, max=in_w - 1)
#     y_indices = torch.clamp(y.unsqueeze(dim=-1) + ky, max=in_h - 1)
#     # x_indices = torch.stack([x, x + 1], dim=-1).clamp(max=in_w - 1)
#     # y_indices = torch.stack([y, y + 1], dim=-1).clamp(max=in_h - 1)

#     dtype = torch.float32 if not input.is_floating_point() else input.dtype
#     x_lambda = torch.clamp(x_f32 - x, min=0.0, max=1.0).to(dtype)
#     # x_lambda shape is [output_size[1], ]
#     # x_weights = torch.stack([1.0 - x_lambda, x_lambda], dim=-1)
#     # x_weights shape is [output_size[1], 2]

#     y_lambda = torch.clamp(y_f32 - y, min=0.0, max=1.0).to(dtype)
#     # y_lambda shape is [output_size[0], ]
#     # y_weights = torch.stack([1.0 - y_lambda, y_lambda], dim=-1)
#     # y_weights shape is [output_size[0], 2]

#     y_indices = y_indices.view(*y_indices.shape, 1, 1)
#     # input_selected = aten._unsafe_index(input, [None, None, y_indices, x_indices])
#     input_selected = input[..., y_indices, x_indices].to(dtype)
#     # input_selected shape is [N, C, output_size[0], 2, output_size[1], 2]

#     # y_weights = y_weights.view(output_size[0], 2, 1, 1)
#     # x_weights = x_weights.view(1, output_size[1], 2)
#     # output = (y_weights * (x_weights * input_selected)).sum(dim=(-1, -3))

#     q = input_selected[..., 0] + torch.mul(input_selected[..., 1] - input_selected[..., 0], x_lambda)
#     print("q:", q.shape)
#     print("y_lambda:", y_lambda.shape, y_lambda.mean())
#     output = q[..., 0, :] + torch.mul(q[..., 1, :] - q[..., 0, :], y_lambda.view(-1, 1))

#     # convert output to correct memory format, if necessary
#     # memory_format = utils.suggest_memory_format(input)
#     memory_format = torch.contiguous_format

#     # following "heuristic: only use channels_last path when it's faster than the contiguous path"
#     if input.device.type == "cuda" and n_channels < 16:
#         memory_format = torch.contiguous_format

#     output = output.contiguous(memory_format=memory_format)

#     if not input.is_floating_point():
#         output = output.round()

#     return output


# --- multiple loads
def upsample_bilinear2d(
    input: Tensor,
    output_size: List[int],
    align_corners: bool,
    scales_h: Optional[float] = None,
    scales_w: Optional[float] = None,
) -> Tensor:
    # get dimensions of original image
    _, n_channels, in_h, in_w = input.shape

    # Calculate horizontal and vertical scaling factor
    h_scale_factor = _compute_scale(in_h, output_size[0], align_corners, scales_h)
    w_scale_factor = _compute_scale(in_w, output_size[1], align_corners, scales_w)

    print(h_scale_factor, w_scale_factor)

    i = torch.arange(output_size[0], device=input.device)
    j = torch.arange(output_size[1], device=input.device)

    x_f32 = _compute_source_index(w_scale_factor, j, align_corners).clamp(min=0.0)
    y_f32 = _compute_source_index(h_scale_factor, i, align_corners).clamp(min=0.0).unsqueeze(-1)
    x = x_f32.to(torch.int64)
    y = y_f32.to(torch.int64)

    # xp1 = (x + 1).clamp(max=in_w - 1).to(torch.int64)
    # yp1 = (y + 1).clamp(max=in_h - 1).to(torch.int64)
    xp1 = torch.where(x < in_w - 1, x + 1, x)
    yp1 = torch.where(y < in_h - 1, y + 1, y)

    v1 = input[..., y, x]
    v2 = input[..., y, xp1]
    v3 = input[..., yp1, x]
    v4 = input[..., yp1, xp1]

    dtype = torch.float32 if not input.is_floating_point() else input.dtype
    if not input.is_floating_point():
        v1 = v1.to(dtype)
        v2 = v2.to(dtype)
        v3 = v3.to(dtype)
        v4 = v4.to(dtype)

    yscale = (y_f32 - y).clamp(0.0, 1.0).to(dtype)
    xscale = (x_f32 - x).clamp(0.0, 1.0).to(dtype)

    # x1 * (1 - alpha) + x2 * alpha == x1 + (x2 - x1) * alpha
    q1 = v1 + torch.mul(v2 - v1, xscale)
    q2 = v3 + torch.mul(v4 - v3, xscale)
    output = q1 + torch.mul(q2 - q1, yscale)

    # convert output to correct memory format, if necessary
    # memory_format = utils.suggest_memory_format(input)
    memory_format = torch.contiguous_format

    # following "heuristic: only use channels_last path when it's faster than the contiguous path"
    if input.device.type == "cuda" and n_channels < 16:
        memory_format = torch.contiguous_format

    output = output.contiguous(memory_format=memory_format)

    if not input.is_floating_point():
        output = output.round()

    return output

def transform(img, osize, align_corners=False):
    img = torch.nn.functional.interpolate(img, size=osize, mode="bilinear", antialias=False, align_corners=align_corners)
    return img


def c_transform(img, osize, align_corners=False):
    out = upsample_bilinear2d(img, osize, align_corners=align_corners)
    if out.dtype != img.dtype:
        out = out.to(img.dtype)
    return out


device = "cpu"

# memory_format = torch.channels_last
memory_format = torch.contiguous_format

torch.manual_seed(12)
# x = torch.randint(0, 256, size=(2, 3, 500, 400), dtype=torch.uint8)
x = torch.randint(0, 256, size=(2, 3, 500, 400), dtype=torch.float32)

# Input (1, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (200, 300)
# x = torch.randint(0, 256, size=(1, 3, 1200, 1300), dtype=torch.uint8)
# x = torch.arange(3 * 345 * 456, device=device).reshape(1, 3, 345, 456).to(torch.uint8)
# x = torch.arange(3 * 345 * 456, device=device).reshape(1, 3, 345, 456).to(torch.uint8)
x = x.to(torch.float32)
x = x.contiguous(memory_format=memory_format)

# isize = (4, 4)
# osize = (3, 3)
# x = torch.rand(2, 3, *isize, device=device)

# osize = (500, 200)
osize = (300, 256)
# osize = (200, 300)

output = c_transform(x, osize)
expected = transform(x, osize)
# expected_f = transform(x.float(), osize)

torch.set_printoptions(precision=6)

print(output.dtype, expected.dtype)
print(output.shape, expected.shape)
print(output.stride(), expected.stride())

print(output[0, 0, :3, :5])
print(expected[0, 0, :3, :5])
# print(expected_f[0, 0, :3, :5])

m = (output.float() - expected.float()).abs() > 1
print("Diff:")
print(output[m][:5])
print(expected[m][:5])

kwargs = {}
if x.dtype == torch.uint8:
    kwargs = {
        "atol": 1.0,
        "rtol": 0.0,
    }

torch.testing.assert_close(output, expected, **kwargs)
