import torch
import torch.utils.benchmark as benchmark

from torch.nn.functional import grid_sample, affine_grid


def transform(img, theta, mode, align_corners):
    n, c, h, w = img.shape
    grid = affine_grid(theta, size=(n, c, h, w), align_corners=align_corners)
    output = grid_sample(img, grid, align_corners=align_corners, mode=mode)
    return output


def main():

    results = []
    min_run_time = 2.0

    a = torch.deg2rad(torch.tensor(45.0))
    ca, sa = torch.cos(a), torch.sin(a)
    s1 = 1.23
    s2 = 1.34

    n, c, h, w = 2, 3, 345, 456

    theta = torch.tensor([[
        [ca / s1, sa, 0.0],
        [-sa, ca / s2, 0.0],
    ]])
    theta = theta.expand(2, 2, 3).contiguous()


    for align_corners in [True, False]:
        for mode in ["bilinear", "nearest", "bicubic"]:

            for device in ["cpu", "cuda"]:
            # for device in ["cuda", ]:
            # for device in ["cpu", ]:

                torch.manual_seed(12)
                num_threads = 1

                torch.set_num_threads(num_threads)
                for memory_format in [torch.contiguous_format, torch.channels_last]:
                # for memory_format in [torch.contiguous_format, ]:
                    for dtype in [torch.float32, ]:

                        x = torch.arange(n * c * h * w, device=device).reshape(n, c, h, w).to(torch.uint8)
                        x = x.to(dtype=dtype)
                        x = x.contiguous(memory_format=memory_format)

                        theta = theta.to(device=device, dtype=dtype)
                        c_transform = torch.compile(transform)

                        output = c_transform(x, theta, mode, align_corners)
                        expected = transform(x, theta, mode, align_corners)
                        # torch.testing.assert_close(output, expected)

                        results.append(
                            benchmark.Timer(
                                stmt=f"fn(x, theta, mode, align_corners)",
                                globals={
                                    "fn": transform,
                                    "x": x,
                                    "theta": theta,
                                    "mode": mode,
                                    "align_corners": align_corners,
                                },
                                num_threads=torch.get_num_threads(),
                                label=f"Affine grid sampling, {device}",
                                sub_label=f"Input: {tuple(x.shape)} {x.dtype}, {memory_format}, align_corners={align_corners}, mode={mode}",
                                description=f"Eager",
                            ).blocked_autorange(min_run_time=min_run_time)
                        )
                        results.append(
                            benchmark.Timer(
                                stmt=f"fn(x, theta, mode, align_corners)",
                                globals={
                                    "fn": c_transform,
                                    "x": x,
                                    "theta": theta,
                                    "mode": mode,
                                    "align_corners": align_corners,
                                },
                                num_threads=torch.get_num_threads(),
                                label=f"Affine grid sampling, {device}",
                                sub_label=f"Input: {tuple(x.shape)} {x.dtype}, {memory_format}, align_corners={align_corners}, mode={mode}",
                                description=f"Compiled",
                            ).blocked_autorange(min_run_time=min_run_time)
                        )

    compare = benchmark.Compare(results)
    compare.print()


if __name__ == "__main__":

    print("")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch config: {torch.__config__.show()}")
    print("")

    main()
