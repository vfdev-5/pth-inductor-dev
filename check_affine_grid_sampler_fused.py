# TORCH_COMPILE_DEBUG=1 python check_affine_grid_sampler_fused.py

import os

import torch

# from torch.nn.functional import grid_sample, affine_grid
# from torch.nn.functional import grid_sample
from torch._decomp.decompositions import Tensor, TensorSequenceType, _sum_tensors, _upsample_cubic_interp1d


if not ("OMP_NUM_THREADS" in os.environ):
    torch.set_num_threads(4)


print("")
print(f"Torch version: {torch.__version__}")
print(f"Torch config: {torch.__config__.show()}")
print("")

torch.set_printoptions(precision=7)


def _linspace_from_neg_one(
    num_steps: int, align_corners: bool, dtype: torch.dtype, device: torch.device
):
    if num_steps <= 1:
        return torch.tensor(0, device=device, dtype=dtype)

    a = ((num_steps - 1) / num_steps) if not align_corners else 1
    return torch.linspace(-a, a, steps=num_steps, device=device, dtype=dtype)


def _grid_sampler_2d_new(
    a: Tensor,
    grid: Tensor,
    interpolation_mode: int = 0,
    padding_mode: int = 0,
    align_corners: bool = False,
) -> Tensor:

    def unnormalize(coords: Tensor, size: int) -> Tensor:
        # Rescale coordinates from [-1, 1] to:
        #   [0, size - 1] if align_corners is True
        #   [-.5, size -.5] if align_corners is False
        mul = (size * 0.5 - 0.5) if align_corners else (size * 0.5)
        ofs = size * 0.5 - 0.5
        return coords * mul + ofs

    # Reflects coordinates until they fall between low and high (inclusive).
    # The bounds are passed as twice their value so that half-integer values
    # can be represented as ints.
    def reflect_coordinates(coords: Tensor, twice_low: int, twice_high: int) -> Tensor:
        if twice_low == twice_high:
            return torch.zeros_like(coords)
        coords_min = twice_low / 2
        coords_span = (twice_high - twice_low) / 2
        coords2 = (coords - coords_min).abs()
        extra = torch.fmod(coords2, coords_span)
        flips = (coords2 / coords_span).floor().to(dtype=torch.int8)
        return torch.where(
            flips & 1 == 0, extra + coords_min, coords_span + coords_min - extra
        )

    def compute_coordinates(coords: Tensor, size: int) -> Tensor:
        if padding_mode == 0:  # Zero
            return coords
        elif padding_mode == 1:  # Borders
            return torch.clamp(coords, 0, size - 1)
        else:  # padding_mode == 2, Reflection
            if align_corners:
                coords_reflected = reflect_coordinates(coords, 0, 2 * (size - 1))
            else:
                coords_reflected = reflect_coordinates(coords, -1, 2 * size - 1)
            return torch.clamp(coords_reflected, 0, size - 1)

    def compute_source_index(coords: Tensor, size: int) -> Tensor:
        coords_un = unnormalize(coords, size)
        return compute_coordinates(coords_un, size)

    N, C, iH, iW = a.shape
    N, C, oH, oW, _ = grid.shape

    def in_bounds_cond(xs: Tensor, ys: Tensor) -> Tensor:
        t_zero = torch.tensor(0, device=xs.device)
        t_iW = torch.tensor(iW, device=xs.device)
        t_iH = torch.tensor(iH, device=xs.device)
        return torch.logical_and(
            t_zero <= xs,
            torch.logical_and(xs < t_iW, torch.logical_and(t_zero <= ys, ys < t_iH))
        )

    N_idx = torch.arange(N, device=a.device).view(N, 1, 1, 1)
    C_idx = torch.arange(C, device=a.device).view(1, C, 1, 1)

    def clip(xs: Tensor, ys: Tensor, ws: Tensor) -> TensorSequenceType:
        cond = in_bounds_cond(xs, ys)
        # To clip to inside valid coordinates, we map the coordinates
        # to (x, y) = (0, 0) and also set the weight to 0
        # We also change the shape of the tensor to the appropriate one for
        # broadcasting with N_idx, C_idx for the purposes of advanced indexing
        return tuple(
            torch.where(cond, t, 0).view(N, C, oH, oW)
            for t in (xs.to(dtype=torch.int64), ys.to(dtype=torch.int64), ws)
        )

    def get_summand(ix: Tensor, iy: Tensor, w) -> Tensor:
        # Perform clipping, index into input tensor and multiply by weight
        idx_x, idx_y, w_ = clip(ix, iy, w)
        return a[N_idx, C_idx, idx_y, idx_x] * w_

    x = grid[..., 0]
    y = grid[..., 1]

    if interpolation_mode == 0:  # Bilinear
        ix = compute_source_index(x, iW)
        iy = compute_source_index(y, iH)

        ix_nw, iy_nw = ix.floor(), iy.floor()
        ix_ne, iy_ne = ix_nw + 1, iy_nw
        ix_sw, iy_sw = ix_nw, iy_nw + 1
        ix_se, iy_se = ix_ne, iy_sw

        w_nw = (ix_se - ix) * (iy_se - iy)
        w_ne = (ix - ix_sw) * (iy_sw - iy)
        w_sw = (ix_ne - ix) * (iy - iy_ne)
        w_se = (ix - ix_nw) * (iy - iy_nw)

        return _sum_tensors(
            get_summand(ix, iy, w)
            for (ix, iy, w) in (
                (ix_nw, iy_nw, w_nw),
                (ix_ne, iy_ne, w_ne),
                (ix_sw, iy_sw, w_sw),
                (ix_se, iy_se, w_se),
            )
        )
    elif interpolation_mode == 1:  # Nearest
        ix = compute_source_index(x, iW)
        iy = compute_source_index(y, iH)

        ix_nearest = ix.round()
        iy_nearest = iy.round()

        return get_summand(ix_nearest, iy_nearest, 1)
    else:  # interpolation_mode == 2, Bicubic
        ix = unnormalize(x, iW)
        iy = unnormalize(y, iH)

        ix_nw = ix.floor()
        iy_nw = iy.floor()

        tx = ix - ix_nw
        ty = iy - iy_nw

        def get_value_bounded(ix: Tensor, iy: Tensor) -> Tensor:
            x = compute_coordinates(ix, iW)
            y = compute_coordinates(iy, iH)
            return get_summand(x, y, 1)

        def get_coeff(ofs: int) -> Tensor:
            iy_ofs = iy_nw + (ofs - 1)
            cs = (
                get_value_bounded(ix_nw - 1, iy_ofs),
                get_value_bounded(ix_nw, iy_ofs),
                get_value_bounded(ix_nw + 1, iy_ofs),
                get_value_bounded(ix_nw + 2, iy_ofs),
            )
            return _upsample_cubic_interp1d(cs, tx)

        coeffs = tuple(get_coeff(ofs) for ofs in range(4))
        return _upsample_cubic_interp1d(coeffs, ty)


def _grid_sampler_2d_old(
    a: Tensor,
    grid: Tensor,
    interpolation_mode: int = 0,
    padding_mode: int = 0,
    align_corners: bool = False,
    _expand_grid: bool = True,
) -> Tensor:
    # This method is a copy of grid_sampler_2d implementation and introduced with additional arg _expand_grid to
    # optionaly expand the input grid for performance reasons.
    # Experimenting locally it was found that compiled CUDA code is accelerated by ~5x
    # and CPU code by ~2x on bicubic mode, if we expand the grid from (N, H, W, 2) into (N, C, H, W, 2)
    # However, this leads to a slowdown around ~0.8x on CPU bilinear mode, channels first.
    # Thus we apply this hack to not expand the grid for this case.

    # torch._check(
    #     interpolation_mode in (0, 1, 2),
    #     lambda: f"Invalid interpolation mode {interpolation_mode}",
    # )
    # torch._check(
    #     padding_mode in (0, 1, 2), lambda: f"Invalid padding mode {padding_mode}"
    # )

    def unnormalize(coords: Tensor, size: int) -> Tensor:
        # Rescale coordinates from [-1, 1] to:
        #   [0, size - 1] if align_corners is True
        #   [-.5, size -.5] if align_corners is False
        mul = (size * 0.5 - 0.5) if align_corners else (size * 0.5)
        ofs = size * 0.5 - 0.5
        return coords * mul + ofs

    # Reflects coordinates until they fall between low and high (inclusive).
    # The bounds are passed as twice their value so that half-integer values
    # can be represented as ints.
    def reflect_coordinates(coords: Tensor, twice_low: int, twice_high: int) -> Tensor:
        if twice_low == twice_high:
            return torch.zeros_like(coords)
        coords_min = twice_low / 2
        coords_span = (twice_high - twice_low) / 2
        coords2 = (coords - coords_min).abs()
        extra = torch.fmod(coords2, coords_span)
        flips = (coords2 / coords_span).floor().to(dtype=torch.int8)
        return torch.where(
            flips & 1 == 0, extra + coords_min, coords_span + coords_min - extra
        )

    def compute_coordinates(coords: Tensor, size: int) -> Tensor:
        if padding_mode == 0:  # Zero
            return coords
        elif padding_mode == 1:  # Borders
            return torch.clamp(coords, 0, size - 1)
        else:  # padding_mode == 2, Reflection
            if align_corners:
                coords_reflected = reflect_coordinates(coords, 0, 2 * (size - 1))
            else:
                coords_reflected = reflect_coordinates(coords, -1, 2 * size - 1)
            return torch.clamp(coords_reflected, 0, size - 1)

    def compute_source_index(coords: Tensor, size: int) -> Tensor:
        coords_un = unnormalize(coords, size)
        return compute_coordinates(coords_un, size)

    N, C, iH, iW = a.shape
    _, oH, oW, two = grid.shape
    assert two == 2

    if _expand_grid:
        # Let's expand grid to [N, C, oH, oW, 2]
        # This allows to generate a single triton cuda kernel instead of two kernels.
        # Two kernels are due source indices, weights have shape (N, 1, oH, oW), xnumel=N*oH*oW
        # and output has shape (N, C, oH, oW), xnumel=N*C*oH*oW
        # Expanding grid to (N, C, oH, oW, two) unifies xnumel to N*C*oH*oW
        grid = grid.view(N, 1, oH, oW, two).expand(N, C, oH, oW, 2)

    def in_bounds_cond(xs: Tensor, ys: Tensor) -> Tensor:
        t_zero = torch.tensor(0, device=xs.device)
        t_iW = torch.tensor(iW, device=xs.device)
        t_iH = torch.tensor(iH, device=xs.device)
        return torch.logical_and(
            t_zero <= xs,
            torch.logical_and(xs < t_iW, torch.logical_and(t_zero <= ys, ys < t_iH))
        )

    N_idx = torch.arange(N, device=a.device).view(N, 1, 1, 1)
    C_idx = torch.arange(C, device=a.device).view(1, C, 1, 1)

    def clip(xs: Tensor, ys: Tensor, ws: Tensor) -> TensorSequenceType:
        cond = in_bounds_cond(xs, ys)
        # To clip to inside valid coordinates, we map the coordinates
        # to (x, y) = (0, 0) and also set the weight to 0
        # We also change the shape of the tensor to the appropriate one for
        # broadcasting with N_idx, C_idx for the purposes of advanced indexing
        c = C if _expand_grid else 1
        return tuple(
            torch.where(cond, t, 0).view(N, c, oH, oW)
            for t in (xs.to(dtype=torch.int64), ys.to(dtype=torch.int64), ws)
        )

    def get_summand(ix: Tensor, iy: Tensor, w) -> Tensor:
        # Perform clipping, index into input tensor and multiply by weight
        idx_x, idx_y, w_ = clip(ix, iy, w)
        return a[N_idx, C_idx, idx_y, idx_x] * w_

    x = grid[..., 0]
    y = grid[..., 1]

    if interpolation_mode == 0:  # Bilinear
        ix = compute_source_index(x, iW)
        iy = compute_source_index(y, iH)

        ix_nw, iy_nw = ix.floor(), iy.floor()
        ix_ne, iy_ne = ix_nw + 1, iy_nw
        ix_sw, iy_sw = ix_nw, iy_nw + 1
        ix_se, iy_se = ix_ne, iy_sw

        w_nw = (ix_se - ix) * (iy_se - iy)
        w_ne = (ix - ix_sw) * (iy_sw - iy)
        w_sw = (ix_ne - ix) * (iy - iy_ne)
        w_se = (ix - ix_nw) * (iy - iy_nw)

        return _sum_tensors(
            get_summand(ix, iy, w)
            for (ix, iy, w) in (
                (ix_nw, iy_nw, w_nw),
                (ix_ne, iy_ne, w_ne),
                (ix_sw, iy_sw, w_sw),
                (ix_se, iy_se, w_se),
            )
        )
    elif interpolation_mode == 1:  # Nearest
        ix = compute_source_index(x, iW)
        iy = compute_source_index(y, iH)

        ix_nearest = ix.round()
        iy_nearest = iy.round()

        return get_summand(ix_nearest, iy_nearest, 1)
    else:  # interpolation_mode == 2, Bicubic
        ix = unnormalize(x, iW)
        iy = unnormalize(y, iH)

        ix_nw = ix.floor()
        iy_nw = iy.floor()

        tx = ix - ix_nw
        ty = iy - iy_nw

        if not _expand_grid:
            tx = tx.unsqueeze(1)
            ty = ty.unsqueeze(1)

        def get_value_bounded(ix: Tensor, iy: Tensor) -> Tensor:
            x = compute_coordinates(ix, iW)
            y = compute_coordinates(iy, iH)
            return get_summand(x, y, 1)

        def get_coeff(ofs: int) -> Tensor:
            iy_ofs = iy_nw + (ofs - 1)
            cs = (
                get_value_bounded(ix_nw - 1, iy_ofs),
                get_value_bounded(ix_nw, iy_ofs),
                get_value_bounded(ix_nw + 1, iy_ofs),
                get_value_bounded(ix_nw + 2, iy_ofs),
            )
            return _upsample_cubic_interp1d(cs, tx)

        coeffs = tuple(get_coeff(ofs) for ofs in range(4))
        return _upsample_cubic_interp1d(coeffs, ty)


def _make_base_grid_4d(theta: Tensor, h: int, w: int, align_corners: bool):
    dtype = theta.dtype
    device = theta.device

    # Using padding and summation generates a single kernel vs using torch.stack where 3 kernels generated
    # corresponding to each individual tensor: grid_x, grid_y, grid_one
    grid_x = _linspace_from_neg_one(w, align_corners, dtype, device).view(1, w, 1)
    grid_y = _linspace_from_neg_one(h, align_corners, dtype, device).view(h, 1, 1)
    grid_one = torch.ones((1, 1, 1), dtype=dtype, device=device)

    # this is just a temporary hack and we should use torch.stack here once #104480 is merged
    grid_x = torch.nn.functional.pad(grid_x, pad=(0, 2), mode="constant", value=0)
    grid_y = torch.nn.functional.pad(grid_y, pad=(1, 1), mode="constant", value=0)
    grid_one = torch.nn.functional.pad(grid_one, pad=(2, 0), mode="constant", value=0)
    return grid_x + grid_y + grid_one


def transform(img, theta, align_corners, mode):
    n, c, h, w = img.shape

    # # grid = affine_grid(theta, size=(n, c, h, w), align_corners=align_corners)
    # dtype = theta.dtype
    # device = theta.device

    # # Using padding and summation generates a single kernel vs using torch.stack where 3 kernels generated
    # # corresponding to each individual tensor: grid_x, grid_y, grid_one
    # grid_x = _linspace_from_neg_one(w, align_corners, dtype, device).view(1, w, 1)
    # grid_y = _linspace_from_neg_one(h, align_corners, dtype, device).view(h, 1, 1)
    # grid_one = torch.ones((1, 1, 1), dtype=dtype, device=device)

    # # this is just a temporary hack and we should use torch.stack here once #104480 is merged
    # grid_x = torch.nn.functional.pad(grid_x, pad=(0, 2), mode="constant", value=0)
    # grid_y = torch.nn.functional.pad(grid_y, pad=(1, 1), mode="constant", value=0)
    # grid_one = torch.nn.functional.pad(grid_one, pad=(2, 0), mode="constant", value=0)
    # base_grid =  grid_x + grid_y + grid_one
    # base_grid = base_grid.view(1, 1, h, w, 3).expand(n, c, h, w, 3)

    # # base_grid shape is (n, c, h, w, 3) and theta shape is (n, 2, 3)
    # # We do manually a matrix multiplication which is faster than mm()
    # # (n, c, h * w, 3, 1) * (n, 1, 1, 3, 2) -> (n, c, h * w, 2)
    # grid = (base_grid.view(n, c, -1, 3, 1) * theta.mT.view(n, 1, 1, 3, 2)).sum(-2)
    # grid = grid.view(n, c, h, w, 2)

    base_grid = _make_base_grid_4d(theta, h, w, align_corners=align_corners)
    # base_grid shape is (h, w, 3) and theta shape is (n, 2, 3)
    # We do manually a matrix multiplication which is faster than mm()
    # (h * w, 3, 1) * (n, 1, 3, 2) -> (n, h * w, 2)
    grid = (base_grid.view(-1, 3, 1) * theta.mT.unsqueeze(1)).sum(-2)
    grid = grid.view(n, h, w, 2)

    output = _grid_sampler_2d_old(img, grid, align_corners=align_corners, interpolation_mode=mode, _expand_grid=True)
    # output = grid_sample(img, grid, align_corners=align_corners, mode=mode)
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
