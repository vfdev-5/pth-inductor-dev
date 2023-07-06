import torch
# import torch.utils.benchmark as benchmark
from time import perf_counter_ns


def benchmark_timer(
    fn, *, args, label, sub_label, description, min_run_time
):
    elapsed = 0.0
    times = []
    while elapsed < min_run_time * 1e9:

        start = perf_counter_ns()
        output = fn(*args)
        stop = perf_counter_ns()
        assert output is not None
        delta = stop - start
        elapsed += delta
        times.append(delta)

    times = torch.tensor(times).float() * 1e-3
    median = times.median().item()
    min, max = torch.aminmax(times)
    mean = torch.mean(times)

    print(
        label, sub_label, description, " : ", median, "µs (", min.item(), max.item(), ")", mean.item(),
    )


@torch.compile
def flip_inductor(x, dim):
    return torch.flip(x, dims=(dim, ))


def n_flip(x, dim):
    o = torch.flip(x, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))
    return o


n_flip_inductor = torch.compile(n_flip)


def main():
    min_run_time = 10

    torch.manual_seed(12)

    # for op in ["HFlip", "VFlip"]:
    for op in ["VFlip", ]:
    # for op in ["HFlip", ]:

        # for mf in ["channels_last", "channels_first"]:
        for mf in ["channels_first", ]:
        # for mf in ["channels_last", ]:
            # for c, dtype in [(1, torch.int), (3, torch.uint8), (1, torch.float32)]:
            # for c, dtype in [(3, torch.uint8), (1, torch.float32)]:
            # for c, dtype in [(3, torch.long), (1, torch.int), (3, torch.uint8), (1, torch.float32), (3, torch.double)]:
            # for c, dtype in [(3, torch.long), (3, torch.uint8), (3, torch.double)]:
            for c, dtype in [
                # (2, torch.uint8),
                (3, torch.uint8),
                # (5, torch.uint8),
                # (8, torch.uint8),
            ]:
            # for c, dtype in [(1, torch.uint8), ]:

                # for size in [255, 256, 257, 519, 520, 521, 711, 712, 713]:
                # for size in [256, 520, 712]:
                # for size in [224, 256, ]:
                for size in [224, ]:

                    # tensor = torch.randint(0, 256, size=(c, size, size), dtype=dtype)
                    # memory_format = torch.channels_last if mf == "channels_last" else torch.contiguous_format
                    # tensor = tensor[None, ...].contiguous(memory_format=memory_format)
                    tensor = torch.randint(0, 256, size=(1, 3, 224, 224), dtype=torch.uint8)

                    dim = -1 if op == "HFlip" else -2

                    output = tensor.flip(dim)
                    output2 = flip_inductor(tensor, dim=dim)
                    torch.testing.assert_close(output, output2)

                    benchmark_timer(
                        # fn=lambda x: n_flip(x, dim),
                        fn=lambda x: x.flip(dim),
                        args=(tensor, ),
                        label=f"{op} measurements",
                        sub_label=f"{c}, {size}, {dtype}, {mf}",
                        description=f"Torch {torch.__version__}",
                        min_run_time=min_run_time,
                    )
                    benchmark_timer(
                        fn=lambda x: flip_inductor(x, dim=dim),
                        args=(tensor, ),
                        label=f"{op} measurements",
                        sub_label=f"{c}, {size}, {dtype}, {mf}",
                        description=f"Torch inductor {torch.__version__}",
                        min_run_time=min_run_time,
                    )


if __name__ == "__main__":

    import os
    if not ("OMP_NUM_THREADS" in os.environ):
        torch.set_num_threads(1)

    print("")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch config: {torch.__config__.show()}")
    print(f"Num threads: {torch.get_num_threads()}")
    print("")

    main()
