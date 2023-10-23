
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


# kernel path: /tmp/torchinductor_root/ft/cfthltppx6jxxp6zy4l2jgr57p2dew4u447evht2zh72o4g3h7zh.py
# Source Nodes: [img], Original ATen: [aten.div]
# img => div
triton_poi_fused_div_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_0', 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 5)
    x0 = xindex % 5
    x2 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.6764705882352942
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tmp6 + tmp2
    tmp8 = tmp7.to(tl.int64)
    tmp9 = tl.full([1], 456, tl.int64)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tmp11 = tmp5 - tmp4
    tmp12 = tmp11 + tmp2
    tmp13 = tmp12.to(tl.int64)
    tmp14 = tl.full([1], 0, tl.int64)
    tmp15 = triton_helpers.maximum(tmp13, tmp14)
    tmp16 = tmp10 - tmp15
    tmp17 = tl.full([1], 5, tl.int64)
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tmp19 = x0
    tmp20 = tmp19 < tmp18
    tmp21 = tmp12.to(tl.int32)
    tmp22 = triton_helpers.maximum(tmp21, tmp14)
    tmp23 = tmp19 + tmp22
    tmp24 = tmp23.to(tl.float32)
    tmp25 = tmp24 - tmp5
    tmp26 = tmp25 + tmp2
    tmp27 = 0.5964912280701754
    tmp28 = tmp26 * tmp27
    tmp29 = tl.abs(tmp28)
    tmp30 = 1.0
    tmp31 = triton_helpers.minimum(tmp29, tmp30)
    tmp32 = tmp30 - tmp31
    tmp33 = 0.0
    tmp34 = tl.where(tmp20, tmp32, tmp33)
    tmp35 = tmp14 < tmp18
    tmp36 = tmp14 + tmp22
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp37 - tmp5
    tmp39 = tmp38 + tmp2
    tmp40 = tmp39 * tmp27
    tmp41 = tl.abs(tmp40)
    tmp42 = triton_helpers.minimum(tmp41, tmp30)
    tmp43 = tmp30 - tmp42
    tmp44 = tl.where(tmp35, tmp43, tmp33)
    tmp45 = tl.full([1], 1, tl.int64)
    tmp46 = tmp45 < tmp18
    tmp47 = tmp45 + tmp22
    tmp48 = tmp47.to(tl.float32)
    tmp49 = tmp48 - tmp5
    tmp50 = tmp49 + tmp2
    tmp51 = tmp50 * tmp27
    tmp52 = tl.abs(tmp51)
    tmp53 = triton_helpers.minimum(tmp52, tmp30)
    tmp54 = tmp30 - tmp53
    tmp55 = tl.where(tmp46, tmp54, tmp33)
    tmp56 = tmp44 + tmp55
    tmp57 = tl.full([1], 2, tl.int64)
    tmp58 = tmp57 < tmp18
    tmp59 = tmp57 + tmp22
    tmp60 = tmp59.to(tl.float32)
    tmp61 = tmp60 - tmp5
    tmp62 = tmp61 + tmp2
    tmp63 = tmp62 * tmp27
    tmp64 = tl.abs(tmp63)
    tmp65 = triton_helpers.minimum(tmp64, tmp30)
    tmp66 = tmp30 - tmp65
    tmp67 = tl.where(tmp58, tmp66, tmp33)
    tmp68 = tmp56 + tmp67
    tmp69 = tl.full([1], 3, tl.int64)
    tmp70 = tmp69 < tmp18
    tmp71 = tmp69 + tmp22
    tmp72 = tmp71.to(tl.float32)
    tmp73 = tmp72 - tmp5
    tmp74 = tmp73 + tmp2
    tmp75 = tmp74 * tmp27
    tmp76 = tl.abs(tmp75)
    tmp77 = triton_helpers.minimum(tmp76, tmp30)
    tmp78 = tmp30 - tmp77
    tmp79 = tl.where(tmp70, tmp78, tmp33)
    tmp80 = tmp68 + tmp79
    tmp81 = tl.full([1], 4, tl.int64)
    tmp82 = tmp81 < tmp18
    tmp83 = tmp81 + tmp22
    tmp84 = tmp83.to(tl.float32)
    tmp85 = tmp84 - tmp5
    tmp86 = tmp85 + tmp2
    tmp87 = tmp86 * tmp27
    tmp88 = tl.abs(tmp87)
    tmp89 = triton_helpers.minimum(tmp88, tmp30)
    tmp90 = tmp30 - tmp89
    tmp91 = tl.where(tmp82, tmp90, tmp33)
    tmp92 = tmp80 + tmp91
    tmp93 = tmp34 / tmp92
    tl.store(out_ptr0 + (x2), tmp93, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_root/55/c55hu5ez34uaw7ulo6owdu4k3npbod3xnyajyuqap5d3gwphu6wn.py
# Source Nodes: [img], Original ATen: [aten.div, aten.index, aten.mul, aten.sum]
# img => div, index, mul_2, sum_2
triton_poi_fused_div_index_mul_sum_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2097152], 
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_index_mul_sum_1', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1126080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 272
    x1 = (xindex // 272)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (5*x0), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (1 + (5*x0)), xmask, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr0 + (2 + (5*x0)), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr0 + (3 + (5*x0)), xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr0 + (4 + (5*x0)), xmask, eviction_policy='evict_last')
    tmp1 = x0
    tmp2 = tmp1.to(tl.float32)
    tmp3 = 0.5
    tmp4 = tmp2 + tmp3
    tmp5 = 1.6764705882352942
    tmp6 = tmp4 * tmp5
    tmp7 = tmp6 - tmp5
    tmp8 = tmp7 + tmp3
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = triton_helpers.maximum(tmp9, tmp10)
    tmp12 = tmp11 + tmp10
    tmp13 = tl.full([1], 455, tl.int64)
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tmp15 = tl.load(in_ptr1 + (tmp14 + (456*x1)), xmask, eviction_policy='evict_last')
    tmp16 = tmp0 * tmp15
    tmp18 = tl.full([1], 1, tl.int64)
    tmp19 = tmp11 + tmp18
    tmp20 = triton_helpers.minimum(tmp19, tmp13)
    tmp21 = tl.load(in_ptr1 + (tmp20 + (456*x1)), xmask, eviction_policy='evict_last')
    tmp22 = tmp17 * tmp21
    tmp23 = tmp16 + tmp22
    tmp25 = tl.full([1], 2, tl.int64)
    tmp26 = tmp11 + tmp25
    tmp27 = triton_helpers.minimum(tmp26, tmp13)
    tmp28 = tl.load(in_ptr1 + (tmp27 + (456*x1)), xmask, eviction_policy='evict_last')
    tmp29 = tmp24 * tmp28
    tmp30 = tmp23 + tmp29
    tmp32 = tl.full([1], 3, tl.int64)
    tmp33 = tmp11 + tmp32
    tmp34 = triton_helpers.minimum(tmp33, tmp13)
    tmp35 = tl.load(in_ptr1 + (tmp34 + (456*x1)), xmask, eviction_policy='evict_last')
    tmp36 = tmp31 * tmp35
    tmp37 = tmp30 + tmp36
    tmp39 = tl.full([1], 4, tl.int64)
    tmp40 = tmp11 + tmp39
    tmp41 = triton_helpers.minimum(tmp40, tmp13)
    tmp42 = tl.load(in_ptr1 + (tmp41 + (456*x1)), xmask, eviction_policy='evict_last')
    tmp43 = tmp38 * tmp42
    tmp44 = tmp37 + tmp43
    tl.store(out_ptr0 + (x2), tmp44, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 3, 345, 456), (471960, 157320, 456, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf2 = empty_strided((272, 5), (5, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [img], Original ATen: [aten.div]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_div_0.run(buf2, 1360, grid=grid(1360), stream=stream0)
        buf3 = empty_strided((4, 3, 345, 272), (281520, 93840, 272, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [img], Original ATen: [aten.div, aten.index, aten.mul, aten.sum]
        triton_poi_fused_div_index_mul_sum_1.run(buf2, arg0_1, buf3, 1126080, grid=grid(1126080), stream=stream0)
        del arg0_1
        return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 3, 345, 456), (471960, 157320, 456, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
