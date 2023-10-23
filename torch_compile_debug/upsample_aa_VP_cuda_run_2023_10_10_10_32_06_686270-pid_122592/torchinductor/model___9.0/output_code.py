
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_root/6k/c6kf23cii2jjwfoeo5y233grjyx5dwi3ygay7qsvhhuurcw5vk4y.py
# Source Nodes: [img], Original ATen: [aten.clone, aten.index, aten.mul, aten.sum]
# img => clone, index, mul_2, sum_2
triton_poi_fused_clone_index_mul_sum_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16, 131072], tile_hint=TileHint.SQUARE,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_index_mul_sum_0', 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 123576
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 456)
    x2 = xindex % 456
    y0 = yindex % 3
    y1 = (yindex // 3)
    x5 = xindex
    y4 = yindex
    tmp0 = x3
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.2730627306273063
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tmp6 + tmp2
    tmp8 = tmp7.to(tl.int64)
    tmp9 = tl.full([1, 1], 345, tl.int64)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tmp11 = tmp5 - tmp4
    tmp12 = tmp11 + tmp2
    tmp13 = tmp12.to(tl.int64)
    tmp14 = tl.full([1, 1], 0, tl.int64)
    tmp15 = triton_helpers.maximum(tmp13, tmp14)
    tmp16 = tmp10 - tmp15
    tmp17 = tl.full([1, 1], 5, tl.int64)
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tmp19 = tmp14 < tmp18
    tmp20 = tmp12.to(tl.int32)
    tmp21 = triton_helpers.maximum(tmp20, tmp14)
    tmp22 = tmp14 + tmp21
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp23 - tmp5
    tmp25 = tmp24 + tmp2
    tmp26 = 0.7855072463768116
    tmp27 = tmp25 * tmp26
    tmp28 = tl.abs(tmp27)
    tmp29 = 1.0
    tmp30 = triton_helpers.minimum(tmp28, tmp29)
    tmp31 = tmp29 - tmp30
    tmp32 = 0.0
    tmp33 = tl.where(tmp19, tmp31, tmp32)
    tmp34 = tl.full([1, 1], 1, tl.int64)
    tmp35 = tmp34 < tmp18
    tmp36 = tmp34 + tmp21
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp37 - tmp5
    tmp39 = tmp38 + tmp2
    tmp40 = tmp39 * tmp26
    tmp41 = tl.abs(tmp40)
    tmp42 = triton_helpers.minimum(tmp41, tmp29)
    tmp43 = tmp29 - tmp42
    tmp44 = tl.where(tmp35, tmp43, tmp32)
    tmp45 = tmp33 + tmp44
    tmp46 = tl.full([1, 1], 2, tl.int64)
    tmp47 = tmp46 < tmp18
    tmp48 = tmp46 + tmp21
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tmp49 - tmp5
    tmp51 = tmp50 + tmp2
    tmp52 = tmp51 * tmp26
    tmp53 = tl.abs(tmp52)
    tmp54 = triton_helpers.minimum(tmp53, tmp29)
    tmp55 = tmp29 - tmp54
    tmp56 = tl.where(tmp47, tmp55, tmp32)
    tmp57 = tmp45 + tmp56
    tmp58 = tl.full([1, 1], 3, tl.int64)
    tmp59 = tmp58 < tmp18
    tmp60 = tmp58 + tmp21
    tmp61 = tmp60.to(tl.float32)
    tmp62 = tmp61 - tmp5
    tmp63 = tmp62 + tmp2
    tmp64 = tmp63 * tmp26
    tmp65 = tl.abs(tmp64)
    tmp66 = triton_helpers.minimum(tmp65, tmp29)
    tmp67 = tmp29 - tmp66
    tmp68 = tl.where(tmp59, tmp67, tmp32)
    tmp69 = tmp57 + tmp68
    tmp70 = tl.full([1, 1], 4, tl.int64)
    tmp71 = tmp70 < tmp18
    tmp72 = tmp70 + tmp21
    tmp73 = tmp72.to(tl.float32)
    tmp74 = tmp73 - tmp5
    tmp75 = tmp74 + tmp2
    tmp76 = tmp75 * tmp26
    tmp77 = tl.abs(tmp76)
    tmp78 = triton_helpers.minimum(tmp77, tmp29)
    tmp79 = tmp29 - tmp78
    tmp80 = tl.where(tmp71, tmp79, tmp32)
    tmp81 = tmp69 + tmp80
    tmp82 = tmp33 / tmp81
    tmp83 = tmp21 + tmp14
    tmp84 = tl.full([1, 1], 344, tl.int64)
    tmp85 = triton_helpers.minimum(tmp83, tmp84)
    tmp86 = tl.load(in_ptr0 + (y0 + (3*x2) + (1368*tmp85) + (471960*y1)), xmask & ymask)
    tmp87 = tmp82 * tmp86
    tmp88 = tmp44 / tmp81
    tmp89 = tmp21 + tmp34
    tmp90 = triton_helpers.minimum(tmp89, tmp84)
    tmp91 = tl.load(in_ptr0 + (y0 + (3*x2) + (1368*tmp90) + (471960*y1)), xmask & ymask)
    tmp92 = tmp88 * tmp91
    tmp93 = tmp87 + tmp92
    tmp94 = tmp56 / tmp81
    tmp95 = tmp21 + tmp46
    tmp96 = triton_helpers.minimum(tmp95, tmp84)
    tmp97 = tl.load(in_ptr0 + (y0 + (3*x2) + (1368*tmp96) + (471960*y1)), xmask & ymask)
    tmp98 = tmp94 * tmp97
    tmp99 = tmp93 + tmp98
    tmp100 = tmp68 / tmp81
    tmp101 = tmp21 + tmp58
    tmp102 = triton_helpers.minimum(tmp101, tmp84)
    tmp103 = tl.load(in_ptr0 + (y0 + (3*x2) + (1368*tmp102) + (471960*y1)), xmask & ymask)
    tmp104 = tmp100 * tmp103
    tmp105 = tmp99 + tmp104
    tmp106 = tmp80 / tmp81
    tmp107 = tmp21 + tmp70
    tmp108 = triton_helpers.minimum(tmp107, tmp84)
    tmp109 = tl.load(in_ptr0 + (y0 + (3*x2) + (1368*tmp108) + (471960*y1)), xmask & ymask)
    tmp110 = tmp106 * tmp109
    tmp111 = tmp105 + tmp110
    tl.store(out_ptr1 + (y0 + (3*x5) + (370728*y1)), tmp111, xmask & ymask)
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
    assert_size_stride(arg0_1, (4, 3, 345, 456), (471960, 1, 1368, 3))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf4 = empty_strided((4, 3, 271, 456), (370728, 1, 1368, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [img], Original ATen: [aten.clone, aten.index, aten.mul, aten.sum]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_clone_index_mul_sum_0.run(arg0_1, buf4, 12, 123576, grid=grid(12, 123576), stream=stream0)
        del arg0_1
        return (buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 3, 345, 456), (471960, 1, 1368, 3), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
