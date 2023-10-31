
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import device, empty, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_root/su/csufm5oqlj2ni67gyamwnar77yngsh2xtwwhmdjfeojdjkc737bv.py
# Source Nodes: [img], Original ATen: [aten._unsafe_index]
# img => _unsafe_index
triton_poi_fused__unsafe_index_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

print("Compile triton_poi_fused__unsafe_index_0")

@pointwise(
    size_hints=[262144],
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_index_0', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 256) % 256
    x0 = xindex % 256
    x2 = (xindex // 65536)
    x4 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 1.953125
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.int32)
    tmp9 = x0
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp11 + tmp4
    tmp13 = 1.5625
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14.to(tl.int32)
    tmp16 = tl.load(in_ptr0 + (tmp15 + (400*tmp8) + (200000*x2)), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x4), tmp16, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 3, 500, 400), (600000, 200000, 400, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((1, 3, 256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [img], Original ATen: [aten._unsafe_index]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__unsafe_index_0.run(arg0_1, buf0, 196608, grid=grid(196608), stream=stream0)
        del arg0_1
        return (buf0, )


def ref_call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 3, 500, 400), (600000, 200000, 400, 1))
    output = torch.nn.functional.interpolate(arg0_1, (256, 256), mode="nearest")
    del arg0_1
    return (output, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 3, 500, 400), (600000, 200000, 400, 1), device='cuda:0', dtype=torch.float32)
    expected = torch.nn.functional.interpolate(arg0_1, (256, 256), mode="nearest")
    output = call([arg0_1])[0]
    print("- Check consistency v0")
    torch.testing.assert_close(expected, output)

    import torch.utils.benchmark as benchmark

    results = []
    min_run_time = 10

    print("- Start benchmarks")
    torch.set_num_threads(1)

    results.append(
        benchmark.Timer(
            stmt=f"fn([x])",
            globals={
                "fn": ref_call,
                "x": arg0_1,
            },
            num_threads=torch.get_num_threads(),
            label=f"Interpolate nearest, cuda",
            sub_label=f"Input (1, 3, 500, 400) -> 256, 256, {arg0_1.dtype}, CF",
            description=f"Eager",
        ).blocked_autorange(min_run_time=min_run_time)
    )
    results.append(
        benchmark.Timer(
            stmt=f"fn([x])",
            globals={
                "fn": call,
                "x": arg0_1,
            },
            num_threads=torch.get_num_threads(),
            label=f"Interpolate nearest, cuda",
            sub_label=f"Input (1, 3, 500, 400) -> 256, 256, {arg0_1.dtype}, CF",
            description=f"Compiled",
        ).blocked_autorange(min_run_time=min_run_time)
    )

    compare = benchmark.Compare(results)
    compare.print()
