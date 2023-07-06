import torch
import torch.utils.benchmark as benchmark

from torch.nn.functional import grid_sample, affine_grid


def transform(img, theta):
    n, c, h, w = img.shape
    grid = affine_grid(theta, size=(1, c, h, w), align_corners=False)
    grid = grid.expand(n, h, w, 2).to(device=img.device, dtype=img.dtype)
    output = grid_sample(img, grid, align_corners=False)
    return output


def main():

    results = []
    min_run_time = 10

    a = torch.deg2rad(torch.tensor(45.0))
    ca, sa = torch.cos(a), torch.sin(a)
    s1 = 1.23
    s2 = 1.34

    for device in ["cpu", "cuda"]:

        torch.manual_seed(12)

        set_num_threads = [4, ]
        if device == "cpu":
            set_num_threads.append(1)

        for num_threads in set_num_threads:

            torch.set_num_threads(num_threads)
            for memory_format in [torch.contiguous_format, torch.channels_last]:
            # for memory_format in [torch.contiguous_format, ]:
                for dtype in [torch.float32, ]:

                    x = torch.arange(3 * 345 * 456, device=device).reshape(1, 3, 345, 456).to(torch.uint8).to(torch.float32)
                    x = x.contiguous(memory_format=memory_format)

                    theta = torch.tensor([[
                        [ca / s1, sa, 0.0],
                        [-sa, ca / s2, 0.0],
                    ]], device=device, dtype=x.dtype)

                    c_transform = torch.compile(transform)

                    output = c_transform(x, theta)
                    expected = transform(x, theta)
                    # torch.testing.assert_close(output, expected)

                    results.append(
                        benchmark.Timer(
                            stmt=f"fn(x, theta)",
                            globals={
                                "fn": transform,
                                "x": x,
                                "theta": theta
                            },
                            num_threads=torch.get_num_threads(),
                            label=f"Affine grid sampling, {device}",
                            sub_label=f"Input: {x.dtype}, {memory_format}",
                            description=f"Eager",
                        ).blocked_autorange(min_run_time=min_run_time)
                    )
                    results.append(
                        benchmark.Timer(
                            stmt=f"fn(x, theta)",
                            globals={
                                "fn": c_transform,
                                "x": x,
                                "theta": theta
                            },
                            num_threads=torch.get_num_threads(),
                            label=f"Affine grid sampling, {device}",
                            sub_label=f"Input: {x.dtype}, {memory_format}",
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




# Decomp: 30/06/2023
# [---------------------- Affine grid sampling, cpu -----------------------]
#                                                      |  Eager  |  Compiled
# 1 threads: ---------------------------------------------------------------
#       Input: torch.float32, torch.contiguous_format  |   3.6   |    3.5
#       Input: torch.float32, torch.channels_last      |   3.7   |    3.6
# 4 threads: ---------------------------------------------------------------
#       Input: torch.float32, torch.contiguous_format  |   3.8   |    3.5
#       Input: torch.float32, torch.channels_last      |   3.6   |    3.5

# Times are in milliseconds (ms).

# [---------------------- Affine grid sampling, cuda ----------------------]
#                                                      |  Eager  |  Compiled
# 4 threads: ---------------------------------------------------------------
#       Input: torch.float32, torch.contiguous_format  |  110.8  |   153.4
#       Input: torch.float32, torch.channels_last      |  110.7  |   154.2

# Times are in microseconds (us).