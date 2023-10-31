import torch
import torch.utils.benchmark as benchmark


def transform(img):
    return img.flip(dims=(-1,))


def main():

    results = []
    min_run_time = 10

    for isize in [(3456, 4567), ]:

        # for bs in [1, ]:
        # for bs in [4, ]:
        for bs in [1, 4]:
        # for bs in [4, 1]:

            # for device in ["cpu", "cuda"]:
            # for device in ["cpu",]:
            for device in ["cuda",]:

                torch.manual_seed(12)
                # for num_threads in [1, 4]:
                for num_threads in [1,]:
                    torch.set_num_threads(num_threads)
                    # for memory_format in [torch.contiguous_format, torch.channels_last]:
                    # for memory_format in [torch.channels_last, torch.contiguous_format]:
                    # for memory_format in [torch.channels_last, ]:
                    for memory_format in [torch.contiguous_format, ]:
                        # for dtype in [torch.uint8, torch.float32]:
                        # for dtype in [torch.uint8, ]:
                        for dtype in [torch.float32, ]:

                            if device == "cuda" and dtype == torch.uint8:
                                continue

                            x = torch.randint(0, 256, size=(bs, 3, *isize), dtype=dtype, device=device)
                            x = x.contiguous(memory_format=memory_format)

                            # c_transform = torch.compile(transform, mode="reduce-overhead")
                            c_transform = torch.compile(transform)
                            for _ in range(5):
                                _ = c_transform(x)
                                _ = transform(x)

                            results.append(
                                benchmark.Timer(
                                    stmt=f"fn(x)",
                                    globals={
                                        "fn": transform,
                                        "x": x,
                                    },
                                    num_threads=torch.get_num_threads(),
                                    label=f"HFlip, {device}",
                                    sub_label=f"Input ({bs}, 3, {isize[0]}, {isize[1]}), {x.dtype}, {memory_format}",
                                    description=f"Eager",
                                ).blocked_autorange(min_run_time=min_run_time)
                            )
                            results.append(
                                benchmark.Timer(
                                    stmt=f"fn(x)",
                                    globals={
                                        "fn": c_transform,
                                        "x": x,
                                    },
                                    num_threads=torch.get_num_threads(),
                                    label=f"HFlip, {device}",
                                    sub_label=f"Input ({bs}, 3, {isize[0]}, {isize[1]}), {x.dtype}, {memory_format}",
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