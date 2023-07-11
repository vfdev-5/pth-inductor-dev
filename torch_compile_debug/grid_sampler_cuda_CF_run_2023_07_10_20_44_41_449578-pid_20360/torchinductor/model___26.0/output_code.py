
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_root/ah/cahrssbtq6tps4ivzvqztxgpiz4gcdx2m6uyrkzrlicmhnqydrlg.py
# Original ATen: aten.affine_grid_generator

# aten.affine_grid_generator => mul_4, sum_1
triton_poi_fused_affine_grid_generator_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 314640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 2)
    x0 = xindex % 2
    x2 = xindex
    tmp43 = tl.load(in_ptr0 + (3*x0), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr0 + (1 + (3*x0)), xmask, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr0 + (2 + (3*x0)), xmask, eviction_policy='evict_last')
    tmp0 = 0
    tmp1 = 1
    tmp2 = tmp0 < tmp1
    tmp3 = x1 % 456
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 228.0
    tmp6 = tmp4 < tmp5
    tmp7 = 0.004385964912280702
    tmp8 = tmp4 * tmp7
    tmp9 = -0.9978070175438597
    tmp10 = tmp8 + tmp9
    tmp11 = 455 + ((-1)*(x1 % 456))
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp12 * tmp7
    tmp14 = 0.9978070175438597
    tmp15 = tmp14 - tmp13
    tmp16 = tl.where(tmp6, tmp10, tmp15)
    tmp17 = tl.where(tmp2, tmp16, 0.0)
    tmp18 = -1
    tmp19 = tmp18 >= tmp0
    tmp20 = tmp18 < tmp1
    tmp21 = tmp19 & tmp20
    tmp22 = (x1 // 456)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = 172.5
    tmp25 = tmp23 < tmp24
    tmp26 = 0.005797101449275362
    tmp27 = tmp23 * tmp26
    tmp28 = -0.9971014492753624
    tmp29 = tmp27 + tmp28
    tmp30 = 344 + ((-1)*(x1 // 456))
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp31 * tmp26
    tmp33 = 0.9971014492753624
    tmp34 = tmp33 - tmp32
    tmp35 = tl.where(tmp25, tmp29, tmp34)
    tmp36 = tl.where(tmp21, tmp35, 0.0)
    tmp37 = tmp17 + tmp36
    tmp38 = -2
    tmp39 = tmp38 >= tmp0
    tmp40 = 1.0
    tmp41 = tl.where(tmp39, tmp40, 0.0)
    tmp42 = tmp37 + tmp41
    tmp44 = tmp42 * tmp43
    tmp45 = tmp1 < tmp1
    tmp46 = tl.where(tmp45, tmp16, 0.0)
    tmp47 = tmp0 >= tmp0
    tmp48 = tmp47 & tmp2
    tmp49 = tl.where(tmp48, tmp35, 0.0)
    tmp50 = tmp46 + tmp49
    tmp51 = tl.where(tmp19, tmp40, 0.0)
    tmp52 = tmp50 + tmp51
    tmp54 = tmp52 * tmp53
    tmp55 = tmp44 + tmp54
    tmp56 = 2
    tmp57 = tmp56 < tmp1
    tmp58 = tl.where(tmp57, tmp16, 0.0)
    tmp59 = tmp1 >= tmp0
    tmp60 = tmp59 & tmp45
    tmp61 = tl.where(tmp60, tmp35, 0.0)
    tmp62 = tmp58 + tmp61
    tmp63 = tl.where(tmp47, tmp40, 0.0)
    tmp64 = tmp62 + tmp63
    tmp66 = tmp64 * tmp65
    tmp67 = tmp55 + tmp66
    tl.store(out_ptr0 + (x2), tmp67, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_root/xy/cxypyt2flczaexnwunmbtqijjqypcia64m53wq77yzzgaspshplb.py
# Original ATen: aten.grid_sampler_2d

# aten.grid_sampler_2d => add_10, add_4, add_5, add_6, add_7, add_8, add_9, convert_element_type_10, convert_element_type_11, floor, floor_1, full_default_10, full_default_11, full_default_12, full_default_3, full_default_6, full_default_9, ge, ge_1, ge_2, ge_3, ge_4, ge_5, ge_6, ge_7, index, index_1, index_2, index_3, logical_and, logical_and_1, logical_and_10, logical_and_11, logical_and_2, logical_and_3, logical_and_4, logical_and_5, logical_and_6, logical_and_7, logical_and_8, logical_and_9, lt_2, lt_3, lt_4, lt_5, lt_6, lt_7, lt_8, lt_9, mul_10, mul_11, mul_12, mul_13, mul_14, mul_5, mul_6, mul_7, mul_8, mul_9, sub_10, sub_11, sub_4, sub_5, sub_6, sub_7, sub_8, sub_9, where_10, where_11, where_12, where_13, where_4, where_7
triton_poi_fused_grid_sampler_2d_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[524288], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 471960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 157320
    x1 = (xindex // 157320)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr0 + (1 + (2*x0)), xmask, eviction_policy='evict_last')
    tmp1 = 228.0
    tmp2 = tmp0 * tmp1
    tmp3 = 227.5
    tmp4 = tmp2 + tmp3
    tmp5 = tl.math.floor(tmp4)
    tmp6 = 0.0
    tmp7 = tmp5 >= tmp6
    tmp8 = 456.0
    tmp9 = tmp5 < tmp8
    tmp11 = 172.5
    tmp12 = tmp10 * tmp11
    tmp13 = 172.0
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.floor(tmp14)
    tmp16 = tmp15 >= tmp6
    tmp17 = 345.0
    tmp18 = tmp15 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tmp9 & tmp19
    tmp21 = tmp7 & tmp20
    tmp22 = tmp15.to(tl.int64)
    tmp23 = 0
    tmp24 = tl.where(tmp21, tmp22, tmp23)
    tmp25 = triton_helpers.promote_to_tensor(tmp24)
    tl.device_assert((0 <= tmp25) & (tmp25 < 345), "index out of bounds: 0 <= tmp25 < 345")
    tmp26 = tmp5.to(tl.int64)
    tmp27 = tl.where(tmp21, tmp26, tmp23)
    tmp28 = triton_helpers.promote_to_tensor(tmp27)
    tl.device_assert((0 <= tmp28) & (tmp28 < 456), "index out of bounds: 0 <= tmp28 < 456")
    tmp29 = tl.load(in_ptr1 + (tmp27 + (456*tmp24) + (157320*x1)), xmask)
    tmp30 = 1.0
    tmp31 = tmp5 + tmp30
    tmp32 = tmp31 - tmp4
    tmp33 = tmp15 + tmp30
    tmp34 = tmp33 - tmp14
    tmp35 = tmp32 * tmp34
    tmp36 = tl.where(tmp21, tmp35, tmp6)
    tmp37 = tmp31 >= tmp6
    tmp38 = tmp31 < tmp8
    tmp39 = tmp38 & tmp19
    tmp40 = tmp37 & tmp39
    tmp41 = tl.where(tmp40, tmp22, tmp23)
    tmp42 = triton_helpers.promote_to_tensor(tmp41)
    tl.device_assert((0 <= tmp42) & (tmp42 < 345), "index out of bounds: 0 <= tmp42 < 345")
    tmp43 = tmp31.to(tl.int64)
    tmp44 = tl.where(tmp40, tmp43, tmp23)
    tmp45 = triton_helpers.promote_to_tensor(tmp44)
    tl.device_assert((0 <= tmp45) & (tmp45 < 456), "index out of bounds: 0 <= tmp45 < 456")
    tmp46 = tl.load(in_ptr1 + (tmp44 + (456*tmp41) + (157320*x1)), xmask)
    tmp47 = tmp4 - tmp5
    tmp48 = tmp47 * tmp34
    tmp49 = tl.where(tmp40, tmp48, tmp6)
    tmp50 = tmp33 >= tmp6
    tmp51 = tmp33 < tmp17
    tmp52 = tmp50 & tmp51
    tmp53 = tmp9 & tmp52
    tmp54 = tmp7 & tmp53
    tmp55 = tmp33.to(tl.int64)
    tmp56 = tl.where(tmp54, tmp55, tmp23)
    tmp57 = triton_helpers.promote_to_tensor(tmp56)
    tl.device_assert((0 <= tmp57) & (tmp57 < 345), "index out of bounds: 0 <= tmp57 < 345")
    tmp58 = tl.where(tmp54, tmp26, tmp23)
    tmp59 = triton_helpers.promote_to_tensor(tmp58)
    tl.device_assert((0 <= tmp59) & (tmp59 < 456), "index out of bounds: 0 <= tmp59 < 456")
    tmp60 = tl.load(in_ptr1 + (tmp58 + (456*tmp56) + (157320*x1)), xmask)
    tmp61 = tmp14 - tmp15
    tmp62 = tmp32 * tmp61
    tmp63 = tl.where(tmp54, tmp62, tmp6)
    tmp64 = tmp38 & tmp52
    tmp65 = tmp37 & tmp64
    tmp66 = tmp47 * tmp61
    tmp67 = tl.where(tmp65, tmp66, tmp6)
    tmp68 = tl.where(tmp65, tmp55, tmp23)
    tmp69 = tl.where(tmp65, tmp43, tmp23)
    tmp70 = tmp29 * tmp36
    tmp71 = tmp46 * tmp49
    tmp72 = tmp70 + tmp71
    tmp73 = tmp60 * tmp63
    tmp74 = tmp72 + tmp73
    tmp75 = triton_helpers.promote_to_tensor(tmp68)
    tl.device_assert((0 <= tmp75) & (tmp75 < 345), "index out of bounds: 0 <= tmp75 < 345")
    tmp76 = triton_helpers.promote_to_tensor(tmp69)
    tl.device_assert((0 <= tmp76) & (tmp76 < 456), "index out of bounds: 0 <= tmp76 < 456")
    tmp77 = tl.load(in_ptr1 + (tmp69 + (456*tmp68) + (157320*x1)), xmask)
    tmp78 = tmp77 * tmp67
    tmp79 = tmp74 + tmp78
    tl.store(in_out_ptr0 + (x2), tmp79, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 3, 345, 456), (471960, 157320, 456, 1))
    assert_size_stride(arg1_1, (1, 2, 3), (6, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf1 = empty_strided((1, 157320, 2), (314640, 2, 1), device='cuda', dtype=torch.float32)
        stream0 = get_cuda_stream(0)
        triton_poi_fused_affine_grid_generator_0.run(arg1_1, buf1, 314640, grid=grid(314640), stream=stream0)
        del arg1_1
        buf10 = empty_strided((1, 3, 345, 456), (471960, 157320, 456, 1), device='cuda', dtype=torch.float32)
        buf11 = buf10; del buf10  # reuse
        triton_poi_fused_grid_sampler_2d_1.run(buf11, buf1, arg0_1, 471960, grid=grid(471960), stream=stream0)
        del arg0_1
        return (buf11, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 3, 345, 456), (471960, 157320, 456, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((1, 2, 3), (6, 3, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.utils import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
