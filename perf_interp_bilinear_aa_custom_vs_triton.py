import torch
import torch.utils.benchmark as benchmark
from torch._inductor.codecache import PyCodeCache


def transform(img, osize):
    img = torch.nn.functional.interpolate(img, size=osize, mode="bilinear", antialias=True, align_corners=False)

    return img


def main():

    results = []
    min_run_time = 10

    # for isize in [(256, 256), (345, 456), ]:
    # for isize in [(345, 456), ]:
    # for isize in [(256, 256), ]:
    for isize in [(3456, 4567), ]:

        # for osize in [(34, 35), ]:
        # for osize in [(123, 124), ]:
        # for osize in [(224, 224), ]:
        # for osize in [(271, 272), ]:
        # for osize in [(345, 272), ]:

        for osize in [(2345, 3456), ]:
        # for osize in [(isize[0] // 2, isize[1] // 2), ]:
        # for osize in [(isize[0] // 5, isize[1] // 5), ]:

            # for bs in [1, ]:
            for bs in [4, ]:
            # for bs in [1, 4]:
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

                                torch.compiler.reset()

                                if device == "cuda" and dtype == torch.uint8:
                                    continue

                                x = torch.randint(0, 256, size=(bs, 3, *isize), dtype=dtype, device=device)
                                x = x.contiguous(memory_format=memory_format)

                                # c_transform = torch.compile(transform, mode="reduce-overhead")
                                c_transform = torch.compile(transform)
                                _ = c_transform(x, osize)
                                _ = transform(x, osize)

                                assert len(PyCodeCache.cache) == 2, f"{[(k, v.__file__) for k, v in PyCodeCache.cache.items()]}"

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

                                call_fn = None

                                for key in PyCodeCache.cache:
                                    mod = PyCodeCache.cache[key]
                                    if "call" in mod.__dict__:
                                        call_fn = mod.__dict__["call"]

                                if call_fn is not None:
                                    results.append(
                                        benchmark.Timer(
                                            stmt=f"fn([x])",
                                            globals={
                                                "fn": call_fn,
                                                "x": x,
                                            },
                                            num_threads=torch.get_num_threads(),
                                            label=f"Interpolate bilinear, AA=true, {device}",
                                            sub_label=f"Input ({bs}, 3, {isize[0]}, {isize[1]}) -> {osize}, {x.dtype}, {memory_format}",
                                            description=f"Just Triton",
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