from typing import Tuple, Optional
import torch
from torch import Tensor
from functools import partial, reduce


print(torch.__version__)
print(torch.__config__.show())


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


# Need this instead of just sum() to keep mypy happy
def _sum_tensors(ts) -> Tensor:
    return reduce(torch.add, ts)


def _upsample_cubic_convolution1(x: Tensor, A: float) -> Tensor:
    return ((A + 2) * x - (A + 3)) * x * x + 1


def _upsample_cubic_convolution2(x: Tensor, A: float) -> Tensor:
    return ((A * x - 5 * A) * x + 8 * A) * x - 4 * A


def _upsample_get_cubic_coefficients(t: Tensor):
    A = -0.75
    return (
        _upsample_cubic_convolution2(t + 1.0, A),
        _upsample_cubic_convolution1(t, A),
        _upsample_cubic_convolution1(1.0 - t, A),
        _upsample_cubic_convolution2(2.0 - t, A),
    )


def _upsample_cubic_interp1d(src, ts: Tensor) -> Tensor:
    coeffs = _upsample_get_cubic_coefficients(ts)
    return _sum_tensors(s * c for (s, c) in zip(src, coeffs))


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
def upsample_bicubic2d_new_uint8(
    input: Tensor,
    output_size: Tuple[int, int],
    align_corners: bool,
    scale_h: Optional[float] = None,
    scale_w: Optional[float] = None,
) -> Tensor:
    # get dimensions of original image
    _, _, in_h, in_w = input.shape

    # Calculate horizontal and vertical scaling factor
    h_scale_factor = _compute_scale(in_h, output_size[0], align_corners, scale_h)
    w_scale_factor = _compute_scale(in_w, output_size[1], align_corners, scale_w)

    # _, dtype = utils.elementwise_dtypes(
    #     input, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    # )
    dtype = torch.float32

    # We have to create arange with int64 dtype and use .to in order to avoid
    # additional kernels creation in inductor and get a perf slowdown
    i = torch.arange(output_size[0], device=input.device).to(dtype=dtype)
    j = torch.arange(output_size[1], device=input.device).to(dtype=dtype)

    x_f32 = _compute_source_index(w_scale_factor, j, align_corners)
    y_f32 = _compute_source_index(h_scale_factor, i, align_corners)
    y_f32 = y_f32.unsqueeze(-1)

    x = x_f32.floor().to(torch.int64)
    y = y_f32.floor().to(torch.int64)

    yscale = (y_f32 - y).clamp(0.0, 1.0).to(dtype)
    xscale = (x_f32 - x).clamp(0.0, 1.0).to(dtype)

    iys_ofs = (y - 1, y, y + 1, y + 2)
    ixs_ofs = (x - 1, x, x + 1, x + 2)

    def load_bounded(ys, xs):
        y_idx = torch.clamp(ys, 0, in_h - 1)
        x_idx = torch.clamp(xs, 0, in_w - 1)
        # v = aten._unsafe_index(input, [None, None, y_idx, x_idx])
        v = input[..., y_idx, x_idx]
        return v

    weights_x = _upsample_get_cubic_coefficients(xscale)
    weights_y = _upsample_get_cubic_coefficients(yscale)

    def _compute_max_and_precision(weights: Tuple[Tensor]) -> float:
        max_weight = max([t.max() for t in weights]).item()
        weights_precision = 0
        for _ in range(22):
            weights_precision += 1
            next_value = int(0.5 + max_weight * (1 << (weights_precision + 1)))
            if next_value >= (1 << 15):
                break
        return weights_precision

    weights_precision_x = _compute_max_and_precision(weights_x)
    weights_precision_y = _compute_max_and_precision(weights_y)

    weights_x = [
        (w * (1 << weights_precision_x) + torch.sign(w) * 0.5).to(torch.int16)
        for w in weights_x
    ]
    weights_y = [
        (w * (1 << weights_precision_y) + torch.sign(w) * 0.5).to(torch.int16)
        for w in weights_y
    ]

    def get_x_interp(y):
        src_x = tuple(load_bounded(y, x_ofs) for x_ofs in ixs_ofs)

        output = _sum_tensors(
            s.to(torch.int32) * c.to(torch.int32) for s, c in zip(src_x, weights_x)
        ) + (1 << (weights_precision_x - 1))

        output = output >> weights_precision_x
        output = torch.clamp(output, 0, 255).to(torch.uint8)
        return output

    coeffs_y = tuple(get_x_interp(y_ofs) for y_ofs in iys_ofs)

    result = _sum_tensors(
        s.to(torch.int32) * c.to(torch.int32) for s, c in zip(coeffs_y, weights_y)
    ) + (1 << (weights_precision_y - 1))
    result = result >> weights_precision_y
    result = torch.clamp(result, 0, 255)
    result = result.to(torch.uint8)

    # convert output to correct memory format, if necessary
    # memory_format = utils.suggest_memory_format(input)
    memory_format = torch.contiguous_format
    result = result.contiguous(memory_format=memory_format)

    return result


def upsample_bicubic2d_new(
    input: Tensor,
    output_size: Tuple[int, int],
    align_corners: bool,
    scale_h: Optional[float] = None,
    scale_w: Optional[float] = None,
) -> Tensor:
    # get dimensions of original image
    _, _, in_h, in_w = input.shape

    # Calculate horizontal and vertical scaling factor
    h_scale_factor = _compute_scale(in_h, output_size[0], align_corners, scale_h)
    w_scale_factor = _compute_scale(in_w, output_size[1], align_corners, scale_w)

    # _, dtype = utils.elementwise_dtypes(
    #     input, type_promotion_kind=utils.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    # )
    dtype = torch.float32

    # We have to create arange with int64 dtype and use .to in order to avoid
    # additional kernels creation in inductor and get a perf slowdown
    i = torch.arange(output_size[0], device=input.device).to(dtype=dtype)
    j = torch.arange(output_size[1], device=input.device).to(dtype=dtype)

    x_f32 = _compute_source_index(w_scale_factor, j, align_corners)
    y_f32 = _compute_source_index(h_scale_factor, i, align_corners)
    y_f32 = y_f32.unsqueeze(-1)

    x = x_f32.floor().to(torch.int64)
    y = y_f32.floor().to(torch.int64)

    yscale = (y_f32 - y).clamp(0.0, 1.0).to(dtype)
    xscale = (x_f32 - x).clamp(0.0, 1.0).to(dtype)

    iys_ofs = (y - 1, y, y + 1, y + 2)
    ixs_ofs = (x - 1, x, x + 1, x + 2)

    def load_bounded(ys, xs):
        y_idx = torch.clamp(ys, 0, in_h - 1)
        x_idx = torch.clamp(xs, 0, in_w - 1)
        # v = aten._unsafe_index(input, [None, None, y_idx, x_idx])
        v = input[..., y_idx, x_idx]
        if not input.is_floating_point():
            v = v.to(dtype)
        return v

    def get_x_interp(y):
        coeffs_x = tuple(load_bounded(y, x_ofs) for x_ofs in ixs_ofs)
        output = _upsample_cubic_interp1d(coeffs_x, xscale)

        # if input.dtype == torch.uint8:
        #     output = torch.clamp(output, 0, 255)

        return output

    coeffs_y = tuple(get_x_interp(y_ofs) for y_ofs in iys_ofs)
    result = _upsample_cubic_interp1d(coeffs_y, yscale)

    # convert output to correct memory format, if necessary
    # memory_format = utils.suggest_memory_format(input)
    memory_format = torch.contiguous_format
    result = result.contiguous(memory_format=memory_format)

    if input.dtype == torch.uint8:
        result = torch.clamp(result.round(), 0, 255)

    return result


def upsample_bicubic2d_old(
    a: Tensor,
    output_size: Tuple[int, int],
    align_corners: bool,
    scale_h: Optional[float] = None,
    scale_w: Optional[float] = None,
) -> Tensor:
    N, C, iH, iW = a.shape
    oH, oW = output_size

    # def compute_scale(in_size, out_size, align_corners, scale=None):
    #     if align_corners:
    #         return (in_size - 1) / (out_size - 1) if out_size > 1 else 0
    #     else:
    #         return 1 / scale if scale is not None and scale > 0 else in_size / out_size

    # def compute_source_index(scale, dst_index, align_corners):
    #     if align_corners:
    #         return scale * dst_index
    #     else:
    #         return scale * (dst_index + 0.5) - 0.5

    height_scale = _compute_scale(iH, oH, align_corners, scale_h)
    width_scale = _compute_scale(iW, oW, align_corners, scale_w)

    N_idx = torch.arange(N, device=a.device).view(N, 1, 1, 1)
    C_idx = torch.arange(C, device=a.device).view(1, C, 1, 1)
    out_y = torch.arange(oH, device=a.device).view((1, 1, oH, 1))
    out_x = torch.arange(oW, device=a.device).view((1, 1, 1, oW))

    real_x = _compute_source_index(width_scale, out_x, align_corners)

    # creal_x = real_x.clamp(min=0.0)
    # m = real_x != creal_x
    # print(real_x[m], creal_x[m])
    # print(torch.argwhere(m))

    in_x = real_x.floor()
    t_x = real_x - in_x
    ix = in_x.to(dtype=torch.int64)

    real_y = _compute_source_index(height_scale, out_y, align_corners)
    in_y = real_y.floor()
    t_y = real_y - in_y
    iy = in_y.to(dtype=torch.int64)

    iys_ofs = (iy - 1, iy, iy + 1, iy + 2)
    ixs_ofs = (ix - 1, ix, ix + 1, ix + 2)

    def load_bounded(ys, xs):
        y_idx = torch.clamp(ys, 0, iH - 1)
        x_idx = torch.clamp(xs, 0, iW - 1)
        v = a[N_idx, C_idx, y_idx, x_idx]
        if not a.is_floating_point():
            v = v.to(torch.float32)
        return v

    def get_x_interp(y):
        coeffs_x = tuple(load_bounded(y, x_ofs) for x_ofs in ixs_ofs)
        return _upsample_cubic_interp1d(coeffs_x, t_x)

    coeffs_y = tuple(get_x_interp(y_ofs) for y_ofs in iys_ofs)
    result = _upsample_cubic_interp1d(coeffs_y, t_y)

    # convert output to correct memory format, if necessary
    # memory_format = utils.suggest_memory_format(a)
    memory_format = torch.contiguous_format
    result = result.contiguous(memory_format=memory_format)

    return result


def transform(img, osize, align_corners=False):
    img = torch.nn.functional.interpolate(img, size=osize, mode="bicubic", antialias=False, align_corners=align_corners)
    return img


def c_transform(img, osize, align_corners=False):
    # out = upsample_bicubic2d_old(img, osize, align_corners=align_corners)

    if img.dtype == torch.uint8:
        out = upsample_bicubic2d_new_uint8(img, osize, align_corners=align_corners)
    else:
        out = upsample_bicubic2d_new(img, osize, align_corners=align_corners)
    if out.dtype != img.dtype:
        out = out.to(img.dtype)
    return out


device = "cpu"
align_corners = False

# memory_format = torch.channels_last
memory_format = torch.contiguous_format

torch.manual_seed(12)
# x = torch.randint(0, 256, size=(2, 3, 500, 400), dtype=torch.uint8)
# x = torch.randint(0, 256, size=(1, 3, 500, 400), dtype=torch.uint8)
x = torch.randint(30, 220, size=(1, 3, 500, 400), dtype=torch.uint8)

# x = torch.randint(0, 256, size=(1, 3, 500, 400), dtype=torch.float32)
# x = torch.randint(0, 256, size=(2, 3, 345, 456), dtype=torch.float32)

# Input (1, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (200, 300)
# x = torch.randint(0, 256, size=(1, 3, 1200, 1300), dtype=torch.uint8)
# x = torch.arange(3 * 345 * 456, device=device).reshape(1, 3, 345, 456).to(torch.uint8)
# x = torch.arange(3 * 345 * 456, device=device).reshape(1, 3, 345, 456).to(torch.uint8)
# x = x.to(torch.float32)

x = x.contiguous(memory_format=memory_format)
x = x.to(device)

# isize = (4, 4)
# osize = (3, 3)
# x = torch.rand(2, 3, *isize, device=device)

# osize = (500, 200)
# osize = (300, 256)
# osize = (200, 300)
# osize = (400, 500)
osize = (224, 224)

# osize = (500, 250)
# osize = (800, 700)


# x = torch.tensor([
#     [ 12.,  13.,  14.,  15.,  16.,  17.,  18.,  19.],
#     [ 60.,  61.,  62.,  63.,  64.,  65.,  66.,  67.],
#     [108., 109., 110., 111., 112., 113., 114., 115.],
#     [156., 157., 158., 159., 160., 161., 162., 163.],
#     [204., 205., 206., 207., 208., 209., 210., 211.],
#     [252., 253., 254., 255.,   0.,   1.,   2.,   3.],
#     [ 44.,  45.,  46.,  47.,  48.,  49.,  50.,  51.],
#     [ 92.,  93.,  94.,  95.,  96.,  97.,  98.,  99.]
# ], dtype=torch.uint8)[None, None, ...]
# osize = (4, 4)


# x = torch.tensor([
#     [ 12.,  13.,  14.,  15.],
#     [108., 109., 110., 111.],
#     [204., 205., 206., 207.],
#     [252., 253., 254., 255.],
# ], dtype=torch.uint8)[None, None, ...]
# osize = (16, 16)


output = c_transform(x, osize, align_corners=align_corners)
expected = transform(x, osize, align_corners=align_corners)
if x.dtype == torch.uint8:
    expected_f = transform(x.float(), osize, align_corners=align_corners).round().clamp(0, 255).to(x.dtype)
else:
    expected_f = expected

torch.set_printoptions(precision=6)

print(output.dtype, expected.dtype)
print(output.shape, expected.shape)
print(output.stride(), expected.stride())


# print("output:", output)
# print("expected:", expected)
# print("expected_f:", expected_f)

# print(output[0, 0, :, :])
# print(expected[0, 0, :, :])
# print(expected_f[0, 0, :3, :5])

# m = (output.float() - expected.float()).abs() > 1
# print("Diff:")
# print(output[m].tolist()[:5])
# print(expected[m].tolist()[:5])

if x.dtype == torch.uint8:
    kwargs = {
        "atol": 1.0,
        "rtol": 0.0,
    }
else:
    kwargs = {
        "atol": 1e-3,
        "rtol": 0.0,
    }

try:
    torch.testing.assert_close(expected, expected_f, **kwargs)
except AssertionError as e:
    print("\n!!! FAILED torch.testing.assert_close(expected, expected_f, **kwargs)")
    print(e)

try:
    torch.testing.assert_close(output, expected_f, **kwargs)
except AssertionError as e:
    print("\n!!! FAILED torch.testing.assert_close(output, expected_f, **kwargs)")
    print(e)

try:
    torch.testing.assert_close(output, expected, **kwargs)
except AssertionError as e:
    print("\n!!! FAILED torch.testing.assert_close(output, expected, **kwargs)")
    print(e)

