import torch
import triton
import torch.utils.benchmark as benchmark

from torch.nn.functional import grid_sample, affine_grid

import fire


def transform(img, theta, mode, align_corners):
    n, c, h, w = img.shape
    grid = affine_grid(theta, size=(n, c, h, w), align_corners=align_corners)
    output = grid_sample(img, grid, align_corners=align_corners, mode=mode)
    return output


atol_rtol_map = {
    ("bilinear", True, torch.contiguous_format, torch.float32, "cpu"): {"atol": 1e-3, "rtol": 0.0},
    ("bilinear", True, torch.channels_last, torch.float32, "cpu"): {"atol": 1e-3, "rtol": 0.0},

    ("bilinear", True, torch.contiguous_format, torch.float32, "cuda"): {"atol": 1e-3, "rtol": 0.0},
    ("bilinear", True, torch.channels_last, torch.float32, "cuda"): {"atol": 1e-3, "rtol": 0.0},

    ("bilinear", True, torch.contiguous_format, torch.float64, "cuda"): {"atol": 1e-2, "rtol": 0.0},
    ("bilinear", True, torch.channels_last, torch.float64, "cuda"): {"atol": 1e-2, "rtol": 0.0},

    ("bilinear", False, torch.contiguous_format, torch.float32, "cpu"): {"atol": 1e-1, "rtol": 0.0},  # !!!!
    ("bilinear", False, torch.channels_last, torch.float32, "cpu"): {"atol": 1e-1, "rtol": 0.0},  # !!!!

    ("bilinear", False, torch.contiguous_format, torch.float32, "cuda"): {"atol": 1e-1, "rtol": 0.0},  # !!!!
    ("bilinear", False, torch.channels_last, torch.float32, "cuda"): {"atol": 1e-1, "rtol": 0.0},  # !!!!

    ("bilinear", False, torch.contiguous_format, torch.float64, "cuda"): {"atol": 1e-2, "rtol": 0.0},  # !!!!
    ("bilinear", False, torch.channels_last, torch.float64, "cuda"): {"atol": 1e-2, "rtol": 0.0},  # !!!!

    ("nearest", False, torch.contiguous_format, torch.float32, "cpu"): {"atol": 1, "rtol": 0.0},  # !!!!
    ("nearest", False, torch.channels_last, torch.float32, "cpu"): {"atol": 1, "rtol": 0.0},  # !!!!

    ("nearest", False, torch.contiguous_format, torch.float32, "cuda"): {"atol": 1, "rtol": 0.0},  # !!!!
    ("nearest", False, torch.channels_last, torch.float32, "cuda"): {"atol": 1, "rtol": 0.0},  # !!!!

    ("bicubic", True, torch.contiguous_format, torch.float32, "cpu"): {"atol": 1e-2, "rtol": 0.0},
    ("bicubic", True, torch.channels_last, torch.float32, "cpu"): {"atol": 1e-2, "rtol": 0.0},

    ("bicubic", True, torch.contiguous_format, torch.float32, "cuda"): {"atol": 2e-2, "rtol": 0.0},
    ("bicubic", True, torch.channels_last, torch.float32, "cuda"): {"atol": 2e-2, "rtol": 0.0},

    ("bicubic", True, torch.contiguous_format, torch.float64, "cuda"): {"atol": 3e-3, "rtol": 0.0},
    ("bicubic", True, torch.channels_last, torch.float64, "cuda"): {"atol": 3e-3, "rtol": 0.0},

    ("bicubic", False, torch.contiguous_format, torch.float32, "cpu"): {"atol": 1e-1, "rtol": 0.0},  # !!!!
    ("bicubic", False, torch.channels_last, torch.float32, "cpu"): {"atol": 1e-1, "rtol": 0.0},  # !!!!

    ("bicubic", False, torch.contiguous_format, torch.float32, "cuda"): {"atol": 3e-2, "rtol": 0.0},  # !!!!
    ("bicubic", False, torch.channels_last, torch.float32, "cuda"): {"atol": 3e-2, "rtol": 0.0},  # !!!!

    ("bicubic", False, torch.contiguous_format, torch.float64, "cuda"): {"atol": 3e-3, "rtol": 0.0},  # !!!!
    ("bicubic", False, torch.channels_last, torch.float64, "cuda"): {"atol": 3e-3, "rtol": 0.0},  # !!!!
}


def check_consistency(debug, mode, align_corners, memory_format, dtype, device, strict, n=2):

    if not strict:
        debug = True

    if debug:
        # d = atol_rtol_map.get((mode, align_corners, memory_format, dtype, device), {})
        # if d.get("atol", 0) > 1e-2:
        #     atol = d["atol"]
        #     warn_msg = f"- atol={atol} is too high!"
        # else:
        #     warn_msg = ""
        print("-", mode, align_corners, memory_format, dtype, device)

    torch.manual_seed(12)

    a = torch.deg2rad(torch.tensor(45.0))
    ca, sa = torch.cos(a), torch.sin(a)
    s1 = 1.23
    s2 = 1.34

    c, h, w = 3, 345, 456

    theta = torch.tensor([[
        [ca / s1, sa, 0.0],
        [-sa, ca / s2, 0.0],
    ]])
    theta = theta.expand(n, 2, 3).contiguous()


    x = torch.arange(n * c * h * w, device=device).reshape(n, c, h, w).to(torch.uint8)
    x = x.to(dtype=dtype)
    x = x.contiguous(memory_format=memory_format)

    theta = theta.to(device=device, dtype=dtype)
    c_transform = torch.compile(transform)

    output = c_transform(x, theta, mode, align_corners)

    expected = transform(x, theta, mode, align_corners)

    if strict:
        torch.testing.assert_close(
            output,
            expected,
        )
    else:
        try:
            torch.testing.assert_close(
                output,
                expected,
            )
        except AssertionError as e:
            msg = str(e)
            print(msg)


def main(
    debug=False,
    strict=True,
):
    print(f"Torch version: {torch.__version__}")
    print(f"Triton: {triton.__version__}")
    print(f"Torch config: {torch.__config__.show()}")
    print(f"Num threads: {torch.get_num_threads()}")
    print("")

    for n in [1, 2]:
        # for device in ["cpu", "cuda"]:
        for device in ["cuda", ]:
            # for mode in ["bilinear", "nearest", "bicubic"]:
            for mode in ["bicubic", ]:
                for align_corners in [True, False]:
                    # for memory_format in [torch.contiguous_format, torch.channels_last]:
                    for memory_format in [torch.torch.channels_last]:
                        # for dtype in [torch.float32, torch.float64]:
                        for dtype in [torch.float64, ]:

                            check_consistency(
                                debug, mode, align_corners, memory_format, dtype, device, strict, n=n
                            )

if __name__ == "__main__":
    fire.Fire(main)
