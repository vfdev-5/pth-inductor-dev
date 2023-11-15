import pickle
from pathlib import Path
import unittest.mock

import torch
import triton
import torch.utils.benchmark as benchmark

# We need to set cache size very large to avoid benchmarking eager mode as compiled
torch._dynamo.config.cache_size_limit = 100000

import fire


def patched_as_column_strings(self):
    concrete_results = [r for r in self._results if r is not None]
    env = f"({concrete_results[0].env})" if self._render_env else ""
    env = env.ljust(self._env_str_len + 4)
    output = ["  " + env + concrete_results[0].as_row_name]
    for m, col in zip(self._results, self._columns or ()):
        if m is None:
            output.append(col.num_to_str(None, 1, None))
        else:
            if len(m.times) == 1:
                spread = 0
            else:
                spread = float(torch.tensor(m.times, dtype=torch.float64).std(unbiased=len(m.times) > 1))
                if col._trim_significant_figures:
                    spread = benchmark.utils.common.trim_sigfig(spread, m.significant_figures)
            output.append(f"{m.median / self._time_scale:>3.3f} (+-{spread / self._time_scale:>3.3f})")
    return output


def transform(img, osize, mode):
    img = torch.nn.functional.interpolate(img, size=osize, mode=mode)

    return img


def run_benchmark(mode, memory_format, dtype, device, tag="", min_run_time=5.0, n=2):
    results = []
    torch.manual_seed(12)

    x = torch.randint(0, 256, size=(bs, 3, *isize), dtype=dtype, device=device)
    x = x.contiguous(memory_format=memory_format)

    c_transform = torch.compile(transform)
    _ = c_transform(x, osize, mode=mode)
    _ = transform(x, osize, mode=mode)

    results.append(
        benchmark.Timer(
            stmt=f"fn(x, osize, mode)",
            globals={
                "fn": transform,
                "x": x,
                "osize": osize,
                "mode": mode,
            },
            num_threads=torch.get_num_threads(),
            label=f"Interpolate {mode}, {device}",
            sub_label=f"Input ({bs}, 3, {isize[0], isize[1]}) -> {osize}, {x.dtype}, {memory_format}",
            description=f"Eager",
        ).blocked_autorange(min_run_time=min_run_time)
    )
    results.append(
        benchmark.Timer(
            stmt=f"fn(x, osize, mode)",
            globals={
                "fn": c_transform,
                "x": x,
                "osize": osize,
                "mode": mode,
            },
            num_threads=torch.get_num_threads(),
            label=f"Interpolate {mode}, {device}",
            sub_label=f"Input ({bs}, 3, {isize[0], isize[1]}) -> {osize}, {x.dtype}, {memory_format}",
            description=f"Compiled",
        ).blocked_autorange(min_run_time=min_run_time)
    )


def main(
    output_folder: str,
    min_run_time: float = 10.0,
    tag: str = "",
    display: bool = True,
    num_threads: int = 1,
):
    torch.set_num_threads(num_threads)
    from datetime import datetime

    now = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_filepath = Path(output_folder) / f"{now}-affine-grid-sampler-{tag}.pkl"

    print(f"Output filepath: {str(output_filepath)}")
    print(f"Torch version: {torch.__version__}")
    print(f"Triton: {triton.__version__}")
    print(f"Torch config: {torch.__config__.show()}")
    print(f"Num threads: {torch.get_num_threads()}")
    print("")

    test_results = []

    for mode in ["nearest", "nearest-exact"]:

        for isize, osize, skip_devices in [
            [(500, 400), (256, 256), ()],
            [(1200, 1300), (200, 300), ()],
            [(300, 400), (600, 700), ()],
            [(2345, 2456), (1234, 1345), ()],
            [(1234, 1345), (2345, 2456), ()],
            [(2345, 2456), (120, 200), ()],
        ]:
                for bs in [1, 4]:
                    for device in ["cpu", "cuda"]:
                        if device in skip_devices:
                            continue
                        for memory_format in [torch.contiguous_format, torch.channels_last]:
                            for dtype in [torch.uint8, torch.float32]:

                                test_results += run_benchmark(
                                    mode, memory_format, dtype, device, tag, min_run_time, n=n
                                )

    with open(output_filepath, "wb") as handler:
        output = {
            "filepath": str(output_filepath),
            "torch_version": torch.__version__,
            "torch_config": torch.__config__.show(),
            "num_threads": torch.get_num_threads(),
            "triton": triton.__version__,
            "test_results": test_results,
        }
        pickle.dump(output, handler)

    if display:
        with unittest.mock.patch(
            "torch.utils.benchmark.utils.compare._Row.as_column_strings", patched_as_column_strings
        ):
            compare = benchmark.Compare(test_results)
            compare.print()


if __name__ == "__main__":
    fire.Fire(main)
