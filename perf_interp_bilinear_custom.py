import torch
import torch.utils.benchmark as benchmark

# We need to set cache size very large to avoid benchmarking eager mode as compiled
torch._dynamo.config.cache_size_limit = 100000


def transform(img, osize, align_corners=False):
    img = torch.nn.functional.interpolate(
        img, size=osize, mode="bilinear", antialias=False, align_corners=align_corners
    )
    return img


def main():

    results = []
    min_run_time = 10

    for isize, osize in [
        [(500, 400), (256, 256)],
        [(1200, 1300), (200, 300)],
        [(300, 400), (600, 700)],
    ]:

        # for bs in [1, 4]:
        for bs in [1, ]:

            # for device in ["cpu", "cuda"]:
            for device in ["cpu", ]:

                torch.manual_seed(12)
                torch.set_num_threads(1)
                for memory_format in [torch.contiguous_format, torch.channels_last]:
                # for memory_format in [torch.contiguous_format, ]:
                    # for dtype in [torch.uint8, torch.float32]:
                    for dtype in [torch.uint8, ]:
                    # for dtype in [torch.float32, ]:

                        for align_corners in [True, False]:
                        # for align_corners in [False, ]:

                            if device == "cuda" and dtype == torch.uint8:
                                continue

                            x = torch.randint(0, 256, size=(bs, 3, *isize), dtype=dtype, device=device)
                            x = x.contiguous(memory_format=memory_format)

                            # c_transform = torch.compile(transform, dynamic=True)
                            c_transform = torch.compile(transform)
                            output = c_transform(x, osize)
                            expected = transform(x, osize)

                            # if x.is_floating_point():
                            #     torch.testing.assert_close(output, expected, atol=5e-3, rtol=0.0)
                            # else:
                            #     torch.testing.assert_close(output.float(), expected.float(), atol=1.0, rtol=0.0)

                            results.append(
                                benchmark.Timer(
                                    stmt=f"fn(x, osize)",
                                    globals={
                                        "fn": transform,
                                        "x": x,
                                        "osize": osize,
                                        "align_corners": align_corners,
                                    },
                                    num_threads=torch.get_num_threads(),
                                    label=f"Interpolate bilinear, AA=false, {device}",
                                    sub_label=f"Input ({bs}, 3, {isize[0]}, {isize[1]}) -> {osize}, {x.dtype}, {memory_format}, ac={align_corners}",
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
                                        "align_corners": align_corners,
                                    },
                                    num_threads=torch.get_num_threads(),
                                    label=f"Interpolate bilinear, AA=false, {device}",
                                    sub_label=f"Input ({bs}, 3, {isize[0]}, {isize[1]}) -> {osize}, {x.dtype}, {memory_format}, ac={align_corners}",
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