# TORCH_COMPILE_DEBUG=1 python debug_perf_interp_bilinear_aa_custom.py

import torch
import torch.utils.benchmark as benchmark

torch._dynamo.config.cache_size_limit = 0
do_bench = False


def transform(img):
    img = torch.nn.functional.interpolate(img, size=(270, 270), mode="bilinear", antialias=True)

    return img


def main():

    results = []
    min_run_time = 5

    for bs in [1, ]:
    # for bs in [1, 4]:

        # for device in ["cpu", "cuda"]:
        for device in ["cpu",]:

            torch.manual_seed(12)
            # for num_threads in [1, 4]:
            for num_threads in [1,]:
                torch.set_num_threads(num_threads)
                for memory_format in [torch.contiguous_format, torch.channels_last]:
                # for memory_format in [torch.channels_last, torch.contiguous_format]:
                # for memory_format in [torch.channels_last, ]:
                # for memory_format in [torch.contiguous_format, ]:
                    for dtype in [torch.uint8, torch.float32]:
                    # for dtype in [torch.uint8, ]:
                    # for dtype in [torch.float32, ]:

                        if device == "cuda" and dtype == torch.uint8:
                            continue

                        x = torch.randint(0, 256, size=(bs, 3, 345, 456), dtype=dtype, device=device)
                        x = x.contiguous(memory_format=memory_format)

                        c_transform = torch.compile(transform)
                        output = c_transform(x)
                        expected = transform(x)

                        if do_bench:
                            results.append(
                                benchmark.Timer(
                                    stmt=f"fn(x)",
                                    globals={
                                        "fn": transform,
                                        "x": x,
                                    },
                                    num_threads=torch.get_num_threads(),
                                    label=f"Interpolate bilinear, AA=true, {device}",
                                    sub_label=f"Input ({bs}, 3, 345, 456) -> (270, 270), {x.dtype}, {memory_format}",
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
                                    label=f"Interpolate bilinear, AA=true, {device}",
                                    sub_label=f"Input ({bs}, 3, 345, 456) -> (270, 270), {x.dtype}, {memory_format}",
                                    description=f"Compiled",
                                ).blocked_autorange(min_run_time=min_run_time)
                            )
    if do_bench:
        compare = benchmark.Compare(results)
        compare.print()


if __name__ == "__main__":

    print("")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch config: {torch.__config__.show()}")
    print("")

    main()