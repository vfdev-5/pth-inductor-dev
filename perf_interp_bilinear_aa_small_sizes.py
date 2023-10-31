import torch
import torch.utils.benchmark as benchmark

# We need to set cache size very large to avoid benchmarking eager mode as compiled
torch._dynamo.config.cache_size_limit = 100000


def transform(img, osize):
    img = torch.nn.functional.interpolate(img, size=osize, mode="bilinear", antialias=True)

    return img


def main():

    results = []
    min_run_time = 10.0
    # min_run_time = 0.01

    for isize, osize in [
        [(500, 400), (256, 256)],
        [(345, 456), (123, 124)],
        [(345, 456), (567, 678)],
    ]:

        for bs in [1, 4]:

            # for device in ["cpu", "cuda"]:
            for device in ["cuda", ]:

                torch.manual_seed(12)
                for num_threads in [1,]:
                    torch.set_num_threads(num_threads)
                    for memory_format in [torch.contiguous_format, torch.channels_last]:
                        for dtype in [torch.uint8, torch.float32]:

                            if device == "cuda" and dtype == torch.uint8:
                                continue

                            x = torch.randint(0, 256, size=(bs, 3, *isize), dtype=dtype, device=device)
                            x = x.contiguous(memory_format=memory_format)

                            c_transform = torch.compile(transform)
                            output = c_transform(x, osize)
                            expected = transform(x, osize)

                            if x.is_floating_point():
                                torch.testing.assert_close(output, expected, atol=5e-3, rtol=0.0)
                            else:
                                torch.testing.assert_close(output.float(), expected.float(), atol=1.0, rtol=0.0)

                            results.append(
                                benchmark.Timer(
                                    stmt=f"fn(x, osize)",
                                    globals={
                                        "fn": transform,
                                        "x": x,
                                        "osize": osize,
                                    },
                                    num_threads=torch.get_num_threads(),
                                    label=f"Interpolate bilinear, AA=true, {device}",
                                    sub_label=f"Input ({bs}, 3, {isize[0]}, {isize[1]}) -> {osize}, {x.dtype}, {memory_format}",
                                    description=f"Eager",
                                ).blocked_autorange(min_run_time=min_run_time)
                            )
                            results.append(
                                benchmark.Timer(
                                    stmt=f"fn(x, osize)",
                                    globals={
                                        "fn": c_transform,
                                        "x": x,
                                        "osize": osize,
                                    },
                                    num_threads=torch.get_num_threads(),
                                    label=f"Interpolate bilinear, AA=true, {device}",
                                    sub_label=f"Input ({bs}, 3, {isize[0]}, {isize[1]}) -> {osize}, {x.dtype}, {memory_format}",
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