import torch
import torch.utils.benchmark as benchmark


@torch.compile
def flip_inductor(x, dim):
    return torch.flip(x, dims=(dim, ))


def main():
    results = []
    min_run_time = 10

    torch.manual_seed(12)

    # for op in ["HFlip", "VFlip"]:
    for op in ["VFlip", ]:

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

                    # Tensor flip
                    results.append(
                        benchmark.Timer(
                            stmt=f"data.flip({dim})",
                            globals={
                                "data": tensor,
                            },
                            num_threads=torch.get_num_threads(),
                            label=f"{op} measurements",
                            sub_label=f"{c}, {size}, {dtype}, {mf}",
                            description=f"Torch {torch.__version__}",
                        ).blocked_autorange(min_run_time=min_run_time)
                    )
                    # Tensor flip inductor
                    results.append(
                        benchmark.Timer(
                            stmt=f"fn(data, dim={dim})",
                            globals={
                                "fn": flip_inductor,
                                "data": tensor,
                            },
                            num_threads=torch.get_num_threads(),
                            label=f"{op} measurements",
                            sub_label=f"{c}, {size}, {dtype}, {mf}",
                            description=f"Torch inductor {torch.__version__}",
                        ).blocked_autorange(min_run_time=min_run_time)
                    )

    compare = benchmark.Compare(results)
    compare.print()


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
