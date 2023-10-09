
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


# kernel path: /tmp/torchinductor_root/mn/cmnuilgcferol45crfibo7asr5pqxn5yd5ej2zso33amejytjjry.py
# Source Nodes: [img], Original ATen: [aten.sum]
# img => sum_1
triton_poi_fused_sum_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512], 
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_0', 'configs': [instance_descriptor(divisible_by_16=(0,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 270
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.6888888888888889
    tmp5 = tmp3 * tmp4
    tmp6 = 1.6888889074325562
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 + tmp2
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 456, tl.int64)
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tmp12 = tmp5 - tmp6
    tmp13 = tmp12 + tmp2
    tmp14 = tmp13.to(tl.int32)
    tmp15 = tl.full([1], 0, tl.int64)
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = tmp11 - tmp16
    tmp18 = triton_helpers.maximum(tmp17, tmp15)
    tmp19 = tl.full([1], 5, tl.int64)
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp15 < tmp20
    tmp22 = tmp15 + tmp16
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp23 - tmp5
    tmp25 = tmp24 + tmp2
    tmp26 = 0.5921052631578947
    tmp27 = tmp25 * tmp26
    tmp28 = tl.abs(tmp27)
    tmp29 = 1.0
    tmp30 = triton_helpers.minimum(tmp28, tmp29)
    tmp31 = tmp29 - tmp30
    tmp32 = 0.0
    tmp33 = tl.where(tmp21, tmp31, tmp32)
    tmp34 = tl.full([1], 1, tl.int64)
    tmp35 = tmp34 < tmp20
    tmp36 = tmp34 + tmp16
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp37 - tmp5
    tmp39 = tmp38 + tmp2
    tmp40 = tmp39 * tmp26
    tmp41 = tl.abs(tmp40)
    tmp42 = triton_helpers.minimum(tmp41, tmp29)
    tmp43 = tmp29 - tmp42
    tmp44 = tl.where(tmp35, tmp43, tmp32)
    tmp45 = tmp33 + tmp44
    tmp46 = tl.full([1], 2, tl.int64)
    tmp47 = tmp46 < tmp20
    tmp48 = tmp46 + tmp16
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tmp49 - tmp5
    tmp51 = tmp50 + tmp2
    tmp52 = tmp51 * tmp26
    tmp53 = tl.abs(tmp52)
    tmp54 = triton_helpers.minimum(tmp53, tmp29)
    tmp55 = tmp29 - tmp54
    tmp56 = tl.where(tmp47, tmp55, tmp32)
    tmp57 = tmp45 + tmp56
    tmp58 = tl.full([1], 3, tl.int64)
    tmp59 = tmp58 < tmp20
    tmp60 = tmp58 + tmp16
    tmp61 = tmp60.to(tl.float32)
    tmp62 = tmp61 - tmp5
    tmp63 = tmp62 + tmp2
    tmp64 = tmp63 * tmp26
    tmp65 = tl.abs(tmp64)
    tmp66 = triton_helpers.minimum(tmp65, tmp29)
    tmp67 = tmp29 - tmp66
    tmp68 = tl.where(tmp59, tmp67, tmp32)
    tmp69 = tmp57 + tmp68
    tmp70 = tl.full([1], 4, tl.int64)
    tmp71 = tmp70 < tmp20
    tmp72 = tmp70 + tmp16
    tmp73 = tmp72.to(tl.float32)
    tmp74 = tmp73 - tmp5
    tmp75 = tmp74 + tmp2
    tmp76 = tmp75 * tmp26
    tmp77 = tl.abs(tmp76)
    tmp78 = triton_helpers.minimum(tmp77, tmp29)
    tmp79 = tmp29 - tmp78
    tmp80 = tl.where(tmp71, tmp79, tmp32)
    tmp81 = tmp69 + tmp80
    tl.store(out_ptr0 + (x0), tmp81, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_root/jv/cjvdhdoxmmwue7jssnuigrpxttul5mmo4iyr3f677c4lwsxnctky.py
# Source Nodes: [img], Original ATen: [aten.sum]
# img => sum_2
triton_poi_fused_sum_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512], 
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_sum_1', 'configs': [instance_descriptor(divisible_by_16=(0,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 270
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.2777777777777777
    tmp5 = tmp3 * tmp4
    tmp6 = 1.2777777910232544
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 + tmp2
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 345, tl.int64)
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tmp12 = tmp5 - tmp6
    tmp13 = tmp12 + tmp2
    tmp14 = tmp13.to(tl.int32)
    tmp15 = tl.full([1], 0, tl.int64)
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = tmp11 - tmp16
    tmp18 = triton_helpers.maximum(tmp17, tmp15)
    tmp19 = tl.full([1], 5, tl.int64)
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp15 < tmp20
    tmp22 = tmp15 + tmp16
    tmp23 = tmp22.to(tl.float32)
    tmp24 = tmp23 - tmp5
    tmp25 = tmp24 + tmp2
    tmp26 = 0.782608695652174
    tmp27 = tmp25 * tmp26
    tmp28 = tl.abs(tmp27)
    tmp29 = 1.0
    tmp30 = triton_helpers.minimum(tmp28, tmp29)
    tmp31 = tmp29 - tmp30
    tmp32 = 0.0
    tmp33 = tl.where(tmp21, tmp31, tmp32)
    tmp34 = tl.full([1], 1, tl.int64)
    tmp35 = tmp34 < tmp20
    tmp36 = tmp34 + tmp16
    tmp37 = tmp36.to(tl.float32)
    tmp38 = tmp37 - tmp5
    tmp39 = tmp38 + tmp2
    tmp40 = tmp39 * tmp26
    tmp41 = tl.abs(tmp40)
    tmp42 = triton_helpers.minimum(tmp41, tmp29)
    tmp43 = tmp29 - tmp42
    tmp44 = tl.where(tmp35, tmp43, tmp32)
    tmp45 = tmp33 + tmp44
    tmp46 = tl.full([1], 2, tl.int64)
    tmp47 = tmp46 < tmp20
    tmp48 = tmp46 + tmp16
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tmp49 - tmp5
    tmp51 = tmp50 + tmp2
    tmp52 = tmp51 * tmp26
    tmp53 = tl.abs(tmp52)
    tmp54 = triton_helpers.minimum(tmp53, tmp29)
    tmp55 = tmp29 - tmp54
    tmp56 = tl.where(tmp47, tmp55, tmp32)
    tmp57 = tmp45 + tmp56
    tmp58 = tl.full([1], 3, tl.int64)
    tmp59 = tmp58 < tmp20
    tmp60 = tmp58 + tmp16
    tmp61 = tmp60.to(tl.float32)
    tmp62 = tmp61 - tmp5
    tmp63 = tmp62 + tmp2
    tmp64 = tmp63 * tmp26
    tmp65 = tl.abs(tmp64)
    tmp66 = triton_helpers.minimum(tmp65, tmp29)
    tmp67 = tmp29 - tmp66
    tmp68 = tl.where(tmp59, tmp67, tmp32)
    tmp69 = tmp57 + tmp68
    tmp70 = tl.full([1], 4, tl.int64)
    tmp71 = tmp70 < tmp20
    tmp72 = tmp70 + tmp16
    tmp73 = tmp72.to(tl.float32)
    tmp74 = tmp73 - tmp5
    tmp75 = tmp74 + tmp2
    tmp76 = tmp75 * tmp26
    tmp77 = tl.abs(tmp76)
    tmp78 = triton_helpers.minimum(tmp77, tmp29)
    tmp79 = tmp29 - tmp78
    tmp80 = tl.where(tmp71, tmp79, tmp32)
    tmp81 = tmp69 + tmp80
    tl.store(out_ptr0 + (x0), tmp81, xmask)
''')


# kernel path: /tmp/torchinductor_root/qm/cqmxeb6os5zqyywrohr32kcbbrprxbm25fvs5s3ms554xoqiypvu.py
# Source Nodes: [img], Original ATen: [aten.div, aten.sum]
# img => div, sum_1
triton_poi_fused_div_sum_2 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_sum_2', 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1350
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 270
    x1 = (xindex // 270)
    x2 = xindex
    tmp37 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.6888888888888889
    tmp5 = tmp3 * tmp4
    tmp6 = 1.6888889074325562
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 + tmp2
    tmp9 = tmp8.to(tl.int64)
    tmp10 = tl.full([1], 456, tl.int64)
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tmp12 = tmp5 - tmp6
    tmp13 = tmp12 + tmp2
    tmp14 = tmp13.to(tl.int64)
    tmp15 = tl.full([1], 0, tl.int64)
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = tmp11 - tmp16
    tmp18 = triton_helpers.maximum(tmp17, tmp15)
    tmp19 = tl.full([1], 5, tl.int64)
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = x1
    tmp22 = tmp21 < tmp20
    tmp23 = tmp13.to(tl.int32)
    tmp24 = triton_helpers.maximum(tmp23, tmp15)
    tmp25 = tmp21 + tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp26 - tmp5
    tmp28 = tmp27 + tmp2
    tmp29 = 0.5921052631578947
    tmp30 = tmp28 * tmp29
    tmp31 = tl.abs(tmp30)
    tmp32 = 1.0
    tmp33 = triton_helpers.minimum(tmp31, tmp32)
    tmp34 = tmp32 - tmp33
    tmp35 = 0.0
    tmp36 = tl.where(tmp22, tmp34, tmp35)
    tmp38 = tmp36 / tmp37
    tl.store(out_ptr0 + (x2), tmp38, xmask)
''')


# kernel path: /tmp/torchinductor_root/ha/chatou3pyzsevdqf52g2qhnzg7q3yhevo6zuvbseef5oypapwunm.py
# Source Nodes: [img], Original ATen: [aten.add, aten.index, aten.mul]
# img => add_12, add_13, add_14, index, index_1, index_2, index_3, mul_3, mul_4, mul_5, mul_6
triton_poi_fused_add_index_mul_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1048576], 
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_index_mul_3', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 558900
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 270
    x1 = (xindex // 270)
    x2 = xindex
    tmp16 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr1 + (270 + x0), xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr1 + (540 + x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr1 + (810 + x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.6888888888888889
    tmp5 = tmp3 * tmp4
    tmp6 = 1.6888889074325562
    tmp7 = tmp5 - tmp6
    tmp8 = tmp7 + tmp2
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = triton_helpers.maximum(tmp9, tmp10)
    tmp12 = tmp11 + tmp10
    tmp13 = tl.full([1], 455, tl.int64)
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tmp15 = tl.load(in_ptr0 + (tmp14 + (456*x1)), xmask)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.full([1], 1, tl.int64)
    tmp19 = tmp11 + tmp18
    tmp20 = triton_helpers.minimum(tmp19, tmp13)
    tmp21 = tl.load(in_ptr0 + (tmp20 + (456*x1)), xmask)
    tmp23 = tmp21 * tmp22
    tmp24 = tmp17 + tmp23
    tmp25 = tl.full([1], 2, tl.int64)
    tmp26 = tmp11 + tmp25
    tmp27 = triton_helpers.minimum(tmp26, tmp13)
    tmp28 = tl.load(in_ptr0 + (tmp27 + (456*x1)), xmask)
    tmp30 = tmp28 * tmp29
    tmp31 = tmp24 + tmp30
    tmp32 = tl.full([1], 3, tl.int64)
    tmp33 = tmp11 + tmp32
    tmp34 = triton_helpers.minimum(tmp33, tmp13)
    tmp35 = tl.load(in_ptr0 + (tmp34 + (456*x1)), xmask)
    tmp37 = tmp35 * tmp36
    tmp38 = tmp31 + tmp37
    tl.store(in_out_ptr0 + (x2), tmp38, xmask)
''')


# kernel path: /tmp/torchinductor_root/sg/csgvplc7ah7rupuev6jr5no5jk6xce5tp6cvmeyre2a4tvutwukh.py
# Source Nodes: [img], Original ATen: [aten.div, aten.sum]
# img => div_1, sum_2
triton_poi_fused_div_sum_4 = async_compile.triton('triton_', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_sum_4', 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1350
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 270
    x1 = (xindex // 270)
    x2 = xindex
    tmp37 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.2777777777777777
    tmp5 = tmp3 * tmp4
    tmp6 = 1.2777777910232544
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7 + tmp2
    tmp9 = tmp8.to(tl.int64)
    tmp10 = tl.full([1], 345, tl.int64)
    tmp11 = triton_helpers.minimum(tmp9, tmp10)
    tmp12 = tmp5 - tmp6
    tmp13 = tmp12 + tmp2
    tmp14 = tmp13.to(tl.int64)
    tmp15 = tl.full([1], 0, tl.int64)
    tmp16 = triton_helpers.maximum(tmp14, tmp15)
    tmp17 = tmp11 - tmp16
    tmp18 = triton_helpers.maximum(tmp17, tmp15)
    tmp19 = tl.full([1], 5, tl.int64)
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = x1
    tmp22 = tmp21 < tmp20
    tmp23 = tmp13.to(tl.int32)
    tmp24 = triton_helpers.maximum(tmp23, tmp15)
    tmp25 = tmp21 + tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp26 - tmp5
    tmp28 = tmp27 + tmp2
    tmp29 = 0.782608695652174
    tmp30 = tmp28 * tmp29
    tmp31 = tl.abs(tmp30)
    tmp32 = 1.0
    tmp33 = triton_helpers.minimum(tmp31, tmp32)
    tmp34 = tmp32 - tmp33
    tmp35 = 0.0
    tmp36 = tl.where(tmp22, tmp34, tmp35)
    tmp38 = tmp36 / tmp37
    tl.store(out_ptr0 + (x2), tmp38, xmask)
''')


# kernel path: /tmp/torchinductor_root/at/catl2ppu7ebdnanttq6vpa7ul7yu6bw7jsbzpcgdisk6n4yyock7.py
# Source Nodes: [img], Original ATen: [aten.add, aten.index, aten.mul, aten.unbind]
# img => add_15, add_28, add_29, add_30, add_31, full_default_2, index_4, index_5, index_6, index_7, index_8, index_9, mul_11, mul_12, mul_13, mul_14, mul_15, mul_7
triton_poi_fused_add_index_mul_unbind_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[524288], 
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_index_mul_unbind_5', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 437400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 270) % 270
    x0 = xindex % 270
    x2 = (xindex // 72900)
    x4 = xindex
    tmp61 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp63 = tl.load(in_ptr2 + (270 + x1), xmask, eviction_policy='evict_last')
    tmp66 = tl.load(in_ptr2 + (540 + x1), xmask, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr2 + (810 + x1), xmask, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr2 + (1080 + x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.2777777777777777
    tmp5 = tmp3 * tmp4
    tmp6 = 1.2777777910232544
    tmp7 = tmp5 - tmp6
    tmp8 = tmp7 + tmp2
    tmp9 = tmp8.to(tl.int32)
    tmp10 = tl.full([1], 0, tl.int64)
    tmp11 = triton_helpers.maximum(tmp9, tmp10)
    tmp12 = tmp11 + tmp10
    tmp13 = tl.full([1], 344, tl.int64)
    tmp14 = triton_helpers.minimum(tmp12, tmp13)
    tmp15 = tl.load(in_ptr0 + (x0 + (270*tmp14) + (93150*x2)), xmask)
    tmp16 = x0
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp17 + tmp2
    tmp19 = 1.6888888888888889
    tmp20 = tmp18 * tmp19
    tmp21 = 1.6888889074325562
    tmp22 = tmp20 - tmp21
    tmp23 = tmp22 + tmp2
    tmp24 = tmp23.to(tl.int32)
    tmp25 = triton_helpers.maximum(tmp24, tmp10)
    tmp26 = tl.full([1], 4, tl.int64)
    tmp27 = tmp25 + tmp26
    tmp28 = tl.full([1], 455, tl.int64)
    tmp29 = triton_helpers.minimum(tmp27, tmp28)
    tmp30 = tl.load(in_ptr1 + (tmp29 + (456*tmp14) + (157320*x2)), xmask)
    tmp31 = 0.0
    tmp32 = tmp30 * tmp31
    tmp33 = tmp15 + tmp32
    tmp34 = tl.full([1], 1, tl.int64)
    tmp35 = tmp11 + tmp34
    tmp36 = triton_helpers.minimum(tmp35, tmp13)
    tmp37 = tl.load(in_ptr0 + (x0 + (270*tmp36) + (93150*x2)), xmask)
    tmp38 = tl.load(in_ptr1 + (tmp29 + (456*tmp36) + (157320*x2)), xmask)
    tmp39 = tmp38 * tmp31
    tmp40 = tmp37 + tmp39
    tmp41 = tl.full([1], 2, tl.int64)
    tmp42 = tmp11 + tmp41
    tmp43 = triton_helpers.minimum(tmp42, tmp13)
    tmp44 = tl.load(in_ptr0 + (x0 + (270*tmp43) + (93150*x2)), xmask)
    tmp45 = tl.load(in_ptr1 + (tmp29 + (456*tmp43) + (157320*x2)), xmask)
    tmp46 = tmp45 * tmp31
    tmp47 = tmp44 + tmp46
    tmp48 = tl.full([1], 3, tl.int64)
    tmp49 = tmp11 + tmp48
    tmp50 = triton_helpers.minimum(tmp49, tmp13)
    tmp51 = tl.load(in_ptr0 + (x0 + (270*tmp50) + (93150*x2)), xmask)
    tmp52 = tl.load(in_ptr1 + (tmp29 + (456*tmp50) + (157320*x2)), xmask)
    tmp53 = tmp52 * tmp31
    tmp54 = tmp51 + tmp53
    tmp55 = tmp11 + tmp26
    tmp56 = triton_helpers.minimum(tmp55, tmp13)
    tmp57 = tl.load(in_ptr0 + (x0 + (270*tmp56) + (93150*x2)), xmask)
    tmp58 = tl.load(in_ptr1 + (tmp29 + (456*tmp56) + (157320*x2)), xmask)
    tmp59 = tmp58 * tmp31
    tmp60 = tmp57 + tmp59
    tmp62 = tmp33 * tmp61
    tmp64 = tmp40 * tmp63
    tmp65 = tmp62 + tmp64
    tmp67 = tmp47 * tmp66
    tmp68 = tmp65 + tmp67
    tmp70 = tmp54 * tmp69
    tmp71 = tmp68 + tmp70
    tmp73 = tmp60 * tmp72
    tmp74 = tmp71 + tmp73
    tl.store(in_out_ptr0 + (x4), tmp74, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (2, 3, 345, 456), (471960, 157320, 456, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf1 = empty_strided((270, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [img], Original ATen: [aten.sum]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_sum_0.run(buf1, 270, grid=grid(270), stream=stream0)
        buf3 = empty_strided((270, ), (1, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [img], Original ATen: [aten.sum]
        triton_poi_fused_sum_1.run(buf3, 270, grid=grid(270), stream=stream0)
        buf4 = empty_strided((5, 270), (270, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [img], Original ATen: [aten.div, aten.sum]
        triton_poi_fused_div_sum_2.run(buf1, buf4, 1350, grid=grid(1350), stream=stream0)
        del buf1
        buf5 = empty_strided((2, 3, 345, 270), (279450, 93150, 270, 1), device='cuda', dtype=torch.float32)
        buf6 = buf5; del buf5  # reuse
        # Source Nodes: [img], Original ATen: [aten.add, aten.index, aten.mul]
        triton_poi_fused_add_index_mul_3.run(buf6, arg0_1, buf4, 558900, grid=grid(558900), stream=stream0)
        buf8 = buf4; del buf4  # reuse
        # Source Nodes: [img], Original ATen: [aten.div, aten.sum]
        triton_poi_fused_div_sum_4.run(buf3, buf8, 1350, grid=grid(1350), stream=stream0)
        del buf3
        buf10 = empty_strided((2, 3, 270, 270), (218700, 72900, 270, 1), device='cuda', dtype=torch.float32)
        buf13 = buf10; del buf10  # reuse
        # Source Nodes: [img], Original ATen: [aten.add, aten.index, aten.mul, aten.unbind]
        triton_poi_fused_add_index_mul_unbind_5.run(buf13, buf6, arg0_1, buf8, 437400, grid=grid(437400), stream=stream0)
        del arg0_1
        return (buf13, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 3, 345, 456), (471960, 157320, 456, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
