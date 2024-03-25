import pickle
from pathlib import Path
import unittest.mock
from typing import Optional
from itertools import product

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


def run_benchmark(
    func, input_kwargs, op_kwargs, tag="", min_run_time=5.0, op_label_as_sublabel=False, compile_kwargs={}
):
    results = []
    torch.manual_seed(12)

    bs = input_kwargs["bs"]
    ishape = input_kwargs["ishape"]
    dtype = input_kwargs["dtype"]
    device = input_kwargs["device"]
    memory_format = input_kwargs["memory_format"]

    x = torch.randint(0, 256, size=(bs, *ishape), dtype=dtype, device=device)
    x = x.contiguous(memory_format=memory_format)

    c_func = torch.compile(func, **compile_kwargs)
    _ = c_func(x, **op_kwargs)
    _ = func(x, **op_kwargs)

    op_label = ", ".join([f"{k}: {str(v)}" for k, v in op_kwargs.items()])

    if not op_label_as_sublabel:
        label = f"Interpolate {op_label}, {device}"
        sub_label = f"Input {(bs, *ishape)}, {x.dtype}, {memory_format}"
    else:
        label = f"Interpolate, {device}"
        sub_label = f"Input {(bs, *ishape)}, {x.dtype}, {memory_format} | {op_label}"

    results.append(
        benchmark.Timer(
            stmt=f"fn(x, **op_kwargs)",
            globals={
                "fn": func,
                "x": x,
                "op_kwargs": op_kwargs,
            },
            num_threads=torch.get_num_threads(),
            label=label,
            sub_label=sub_label,
            description=f"Eager ({torch.__version__}) {tag}",
        ).blocked_autorange(min_run_time=min_run_time)
    )
    results.append(
        benchmark.Timer(
            stmt=f"fn(x, **op_kwargs)",
            globals={
                "fn": c_func,
                "x": x,
                "op_kwargs": op_kwargs,
            },
            num_threads=torch.get_num_threads(),
            label=label,
            sub_label=sub_label,
            description=f"Compiled ({torch.__version__}) {tag}",
        ).blocked_autorange(min_run_time=min_run_time)
    )
    return results


def main(
    output_folder: str,
    min_run_time: float = 10.0,
    tag: str = "",
    display: bool = True,
    num_threads: int = 1,
    mode: Optional[str] = None,
    antialias: bool = False,
):
    torch.set_num_threads(num_threads)
    from datetime import datetime

    if mode is None:
        modes = ["nearest", "nearest-exact"]
    else:
        modes = [mode, ]

    now = datetime.now().strftime('%Y%m%d-%H%M%S')
    output_filepath = Path(output_folder) / f"{now}-upsample-{'-'.join(modes)}-{tag}.pkl"

    print(f"Output filepath: {str(output_filepath)}")
    print(f"Torch version: {torch.__version__}")
    print(f"Triton: {triton.__version__}")
    print(f"Torch config: {torch.__config__.show()}")
    print(f"Num threads: {torch.get_num_threads()}")
    print("")

    test_results = []

    input_test_cases = {
        "device": ["cpu", "cuda"],
        "bs": [1, 4],
        "dtype": [torch.uint8, torch.float32],
        "memory_format": [torch.contiguous_format, torch.channels_last],
    }

    op_test_cases = {
        "mode": modes,
        "align_corners": [None, ],
    }

    if 0 < sum(["nearest" in m for m in modes]) < len(modes):
        raise ValueError(f"Can't mix nearest mode with other modes. modes={modes}")

    if not all(["nearest" in m for m in modes]):
        op_test_cases.update({
            "align_corners": [True, False],
            "antialias": [antialias, ],
        })

    compile_kwargs = {
        "fullgraph": True,
        "dynamic": True,
    }

    test_cases = list(input_test_cases.values()) + list(op_test_cases.values())
    test_keys = list(input_test_cases.keys()) + list(op_test_cases.keys())

    def transform(x, *, osize, mode, align_corners, antialias=antialias):
        return torch.nn.functional.interpolate(x, size=osize, mode=mode, align_corners=align_corners, antialias=antialias)

    for isize, osize, skip_devices in [
        [(500, 400), (256, 256), ("cuda", )],
        [(1200, 1300), (200, 300), ("cuda", )],
        [(300, 400), (600, 700), ("cuda", )],
        [(2345, 2456), (1234, 1345), ("cpu", )],
        [(1234, 1345), (2345, 2456), ("cpu", )],
    ]:
        for values in product(*test_cases):
            kwargs = {k:v for k, v in zip(test_keys, values)}
            if kwargs["device"] == "cuda" and kwargs["dtype"] == torch.uint8:
                continue

            if kwargs["device"] in skip_devices:
                continue

            input_kwargs = {k: kwargs[k] for k in kwargs if k in input_test_cases}
            op_kwargs = {k: kwargs[k] for k in kwargs if k in op_test_cases}

            input_kwargs["ishape"] = (3, *isize)
            op_kwargs["osize"] = osize

            print("-", input_kwargs, op_kwargs)
            test_results += run_benchmark(
                transform, input_kwargs, op_kwargs, tag, min_run_time, op_label_as_sublabel=True, compile_kwargs=compile_kwargs
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
            print()
            compare = benchmark.Compare(test_results)
            compare.print()


if __name__ == "__main__":
    fire.Fire(main)
