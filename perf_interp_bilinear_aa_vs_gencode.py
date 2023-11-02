import torch
import torch.utils.benchmark as benchmark
from torch._inductor.codecache import PyCodeCache

# We need to set cache size very large to avoid benchmarking eager mode as compiled
torch._dynamo.config.cache_size_limit = 100000


def transform(img, osize):
    img = torch.nn.functional.interpolate(img, size=osize, mode="bilinear", antialias=True)

    return img


def main():

    results = []
    min_run_time = 10.0
    # min_run_time = 0.001

    for isize, osize, skip_devices in [
        [(500, 400), (256, 256), ()],
        [(1200, 1300), (200, 300), ()],
        [(300, 400), (600, 700), ()],
        [(2345, 2456), (1234, 1345), ("cpu", )],
        [(1234, 1345), (2345, 2456), ("cpu", )],
        [(2345, 2456), (120, 200),("cpu", )],
    ]:

        for bs in [1, 4]:

            for device in ["cpu", "cuda"]:
            # for device in ["cuda", ]:
                if device in skip_devices:
                    continue

                torch.manual_seed(12)
                for num_threads in [1,]:
                    torch.set_num_threads(num_threads)
                    for memory_format in [torch.contiguous_format, torch.channels_last]:
                        for dtype in [torch.uint8, torch.float32]:

                            torch.compiler.reset()
                            PyCodeCache.clear()

                            if device == "cuda" and dtype == torch.uint8:
                                continue

                            x = torch.randint(0, 256, size=(bs, 3, *isize), dtype=dtype, device=device)
                            x = x.contiguous(memory_format=memory_format)

                            c_transform = torch.compile(transform)
                            output = c_transform(x, osize)
                            expected = transform(x, osize)

                            for _ in range(10):
                                _ = c_transform(x, osize)

                            import time
                            time.sleep(2)

                            print(isize, osize, bs, device, memory_format, dtype)
                            # assert len(PyCodeCache.cache) == (2 if device == "cuda" else 1), f"{[(k, v.__file__) for k, v in PyCodeCache.cache.items()]}"
                            call_fn = None
                            found_call_fns = []
                            for key in PyCodeCache.cache:
                                mod = PyCodeCache.cache[key]
                                if "call" in mod.__dict__:
                                    found_call_fns.append(mod.__dict__["call"])

                            assert len(found_call_fns) == 1, (found_call_fns, f"{[(k, v.__file__) for k, v in PyCodeCache.cache.items()]}")
                            call_fn = found_call_fns[0]
                            assert call_fn is not None

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
                                        description=("Just Triton" if device == "cuda" else "Just C++"),
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