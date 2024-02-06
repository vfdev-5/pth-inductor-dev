from typing import Tuple, Optional
import torch
from torch import Tensor
from functools import partial, reduce


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
    print("x:", x)
    print("weights_x:", weights_x[0].shape, weights_x[1].shape, weights_x[2].shape, weights_x[3].shape)
    weights_y = _upsample_get_cubic_coefficients(yscale)

    print("1 weights_x:", weights_x[0][:10], weights_x[1][:10], weights_x[2][:10], weights_x[3][:10])

    def _compute_max_and_precision(weights: Tuple[Tensor]) -> Tuple[float, float]:
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

    print("weights_precision_x:", weights_precision_x)
    print("weights_precision_y:", weights_precision_y)

    weights_x = [
        (w * (1 << weights_precision_x) + torch.sign(w) * 0.5).to(torch.int16)
        for w in weights_x
    ]
    weights_y = [
        (w * (1 << weights_precision_y) + torch.sign(w) * 0.5).to(torch.int16)
        for w in weights_y
    ]

    print("2 weights_x:", weights_x[0][:10], weights_x[1][:10], weights_x[2][:10], weights_x[3][:10])

    def get_x_interp(y):
        src_x = tuple(load_bounded(y, x_ofs) for x_ofs in ixs_ofs)

        # print("src_x:", src_x[0].dtype, src_x[0].shape)
        output = _sum_tensors(
            s.to(torch.int32) * c.to(torch.int32) for s, c in zip(src_x, weights_x)
        ) + (1 << (weights_precision_x - 1))

        output = output >> weights_precision_x
        print("1 output: ", output.dtype, output[0, 0, :5, :5])
        output = torch.clamp(output, 0, 255).to(torch.uint8)
        # print("2 output: ", output[0, 0, :5, :5])
        return output

    coeffs_y = tuple(get_x_interp(y_ofs) for y_ofs in iys_ofs)
    result = coeffs_y[1]

    # # print("coeffs_y:", coeffs_y[0].shape, coeffs_y[0].dtype, coeffs_y[0][0, 0, :5, :5])
    # result = _sum_tensors(
    #     s * c for s, c in zip(coeffs_y, weights_y)
    # ) + (1 << (weights_precision_y - 1))
    # # print("1 result:", result.shape, result.dtype, result[0, 0, :5, :5])
    # result = result >> weights_precision_y
    # # print("2 result:", result.shape, result.dtype, result[0, 0, :5, :5])
    # result = torch.clamp(result, 0, 255)
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
# x = torch.randint(0, 256, size=(1, 3, 1, 20), dtype=torch.uint8)
# x = x[0, 2, ...][None, None, ...].contiguous()
# print(x.tolist())
x = torch.tensor(
    [[[[
        212, 134, 123, 109, 73, 173, 158, 240, 160, 110, 27, 187, 217, 187, 216, 79, 25, 246, 225, 56
    ]]]],
    dtype=torch.uint8
)

osize = (1, 100)

# !!! FAILED torch.testing.assert_close(expected, expected_f, **kwargs)
# Tensor-likes are not close!
# Mismatched elements: 10 / 100 (10.0%)
# Greatest absolute difference: 12 at index (0, 0, 0, 99) (up to 1.0 allowed)
# Greatest relative difference: 0.31578946113586426 at index (0, 0, 0, 99) (up to 0.0 allowed)

# !!! FAILED torch.testing.assert_close(output, expected, **kwargs)
# Tensor-likes are not close!
# Mismatched elements: 10 / 100 (10.0%)
# Greatest absolute difference: 12 at index (0, 0, 0, 99) (up to 1.0 allowed)
# Greatest relative difference: 0.4615384638309479 at index (0, 0, 0, 99) (up to 0.0 allowed)

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


print("output:", output)
print("expected:", expected)
# print("expected_f:", expected_f)
print("Diff", (output != expected).to(torch.uint8))

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



