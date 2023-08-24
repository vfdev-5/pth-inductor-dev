
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


# kernel path: /tmp/torchinductor_root/rb/crbs2hdmkhsbkviyra43p44ohmnzebdub627ysebi3g7h426kjfi.py
# Source Nodes: [affine_grid], Original ATen: [aten.affine_grid_generator]
# affine_grid => mul_4, sum_1
triton_poi_fused_affine_grid_generator_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 629280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 2) % 157320
    x0 = xindex % 2
    x2 = (xindex // 314640)
    x3 = xindex
    tmp43 = tl.load(in_ptr0 + ((3*x0) + (6*x2)), xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr0 + (1 + (3*x0) + (6*x2)), xmask, eviction_policy='evict_last')
    tmp65 = tl.load(in_ptr0 + (2 + (3*x0) + (6*x2)), xmask, eviction_policy='evict_last')
    tmp0 = tl.full([1], 0, tl.int64)
    tmp1 = tl.full([1], 1, tl.int64)
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
    tmp18 = tl.full([1], -1, tl.int64)
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
    tmp38 = tl.full([1], -2, tl.int64)
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
    tmp56 = tl.full([1], 2, tl.int64)
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
    tl.store(out_ptr0 + (x3), tmp67, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_root/qp/cqpcwr4azyqrmxm5h2lcmyxixss2bzvtxvz36ppocxeyyirejbps.py
# Source Nodes: [grid_sample], Original ATen: [aten.grid_sampler_2d]
# grid_sample => add_10, add_11, add_12, add_13, add_14, add_15, add_16, add_17, add_18, add_19, add_20, add_21, add_22, add_23, add_24, add_25, add_26, add_27, add_28, add_29, add_30, add_31, add_32, add_33, add_34, add_35, add_36, add_37, add_38, add_39, add_4, add_40, add_41, add_42, add_43, add_44, add_45, add_46, add_47, add_48, add_49, add_5, add_50, add_51, add_52, add_53, add_54, add_55, add_56, add_57, add_6, add_7, add_8, add_9, convert_element_type_10, convert_element_type_11, convert_element_type_12, convert_element_type_13, convert_element_type_16, convert_element_type_17, convert_element_type_18, convert_element_type_19, convert_element_type_20, convert_element_type_21, convert_element_type_24, convert_element_type_25, convert_element_type_26, convert_element_type_27, convert_element_type_28, convert_element_type_29, convert_element_type_32, convert_element_type_33, convert_element_type_34, convert_element_type_35, convert_element_type_4, convert_element_type_5, convert_element_type_8, convert_element_type_9, floor, floor_1, full_default_1, full_default_10, full_default_11, full_default_12, full_default_13, full_default_14, full_default_15, full_default_16, full_default_17, full_default_18, full_default_19, full_default_2, full_default_20, full_default_23, full_default_24, full_default_25, full_default_26, full_default_27, full_default_28, full_default_29, full_default_3, full_default_30, full_default_31, full_default_32, full_default_33, full_default_34, full_default_35, full_default_36, full_default_39, full_default_4, full_default_40, full_default_41, full_default_42, full_default_43, full_default_44, full_default_45, full_default_46, full_default_47, full_default_48, full_default_49, full_default_50, full_default_51, full_default_52, full_default_55, full_default_56, full_default_57, full_default_58, full_default_59, full_default_60, full_default_61, full_default_62, full_default_63, full_default_64, full_default_7, full_default_8, full_default_9, ge, ge_1, ge_10, ge_11, ge_12, ge_13, ge_14, ge_15, ge_16, ge_17, ge_18, ge_19, ge_2, ge_20, ge_21, ge_22, ge_23, ge_24, ge_25, ge_26, ge_27, ge_28, ge_29, ge_3, ge_30, ge_31, ge_4, ge_5, ge_6, ge_7, ge_8, ge_9, index, index_1, index_10, index_11, index_12, index_13, index_14, index_15, index_2, index_3, index_4, index_5, index_6, index_7, index_8, index_9, logical_and, logical_and_1, logical_and_10, logical_and_11, logical_and_12, logical_and_13, logical_and_14, logical_and_15, logical_and_16, logical_and_17, logical_and_18, logical_and_19, logical_and_2, logical_and_20, logical_and_21, logical_and_22, logical_and_23, logical_and_24, logical_and_25, logical_and_26, logical_and_27, logical_and_28, logical_and_29, logical_and_3, logical_and_30, logical_and_31, logical_and_32, logical_and_33, logical_and_34, logical_and_35, logical_and_36, logical_and_37, logical_and_38, logical_and_39, logical_and_4, logical_and_40, logical_and_41, logical_and_42, logical_and_43, logical_and_44, logical_and_45, logical_and_46, logical_and_47, logical_and_5, logical_and_6, logical_and_7, logical_and_8, logical_and_9, lt_10, lt_11, lt_12, lt_13, lt_14, lt_15, lt_16, lt_17, lt_18, lt_19, lt_2, lt_20, lt_21, lt_22, lt_23, lt_24, lt_25, lt_26, lt_27, lt_28, lt_29, lt_3, lt_30, lt_31, lt_32, lt_33, lt_4, lt_5, lt_6, lt_7, lt_8, lt_9, mul_10, mul_100, mul_101, mul_102, mul_11, mul_12, mul_13, mul_14, mul_15, mul_16, mul_17, mul_18, mul_19, mul_20, mul_21, mul_22, mul_23, mul_24, mul_25, mul_26, mul_27, mul_28, mul_29, mul_30, mul_31, mul_32, mul_33, mul_34, mul_35, mul_36, mul_37, mul_38, mul_39, mul_40, mul_41, mul_42, mul_43, mul_44, mul_45, mul_46, mul_47, mul_48, mul_49, mul_5, mul_50, mul_51, mul_52, mul_53, mul_54, mul_55, mul_56, mul_57, mul_58, mul_59, mul_6, mul_60, mul_61, mul_62, mul_63, mul_64, mul_65, mul_66, mul_67, mul_68, mul_69, mul_7, mul_70, mul_71, mul_72, mul_73, mul_74, mul_75, mul_76, mul_77, mul_78, mul_79, mul_8, mul_80, mul_81, mul_82, mul_83, mul_84, mul_85, mul_86, mul_87, mul_88, mul_89, mul_9, mul_90, mul_91, mul_92, mul_93, mul_94, mul_95, mul_96, mul_97, mul_98, mul_99, sub_10, sub_11, sub_12, sub_13, sub_14, sub_15, sub_16, sub_17, sub_18, sub_19, sub_20, sub_21, sub_22, sub_23, sub_24, sub_25, sub_26, sub_27, sub_28, sub_29, sub_30, sub_31, sub_32, sub_33, sub_34, sub_35, sub_36, sub_37, sub_38, sub_39, sub_4, sub_40, sub_41, sub_42, sub_43, sub_44, sub_45, sub_46, sub_47, sub_48, sub_49, sub_5, sub_6, sub_7, sub_8, sub_9, where_10, where_11, where_12, where_13, where_14, where_15, where_16, where_19, where_2, where_20, where_21, where_22, where_23, where_24, where_25, where_26, where_27, where_28, where_3, where_31, where_32, where_33, where_34, where_35, where_36, where_37, where_38, where_39, where_4, where_40, where_43, where_44, where_45, where_46, where_47, where_48, where_49, where_7, where_8, where_9
triton_poi_fused_grid_sampler_2d_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr1'], 'autotune_hints': set(), 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr1, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 943920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 157320
    x2 = (xindex // 471960)
    x3 = xindex
    x1 = (xindex // 157320) % 3
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (314640*x2)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr0 + (1 + (2*x0) + (314640*x2)), xmask, eviction_policy='evict_last')
    tmp1 = 228.0
    tmp2 = tmp0 * tmp1
    tmp3 = 227.5
    tmp4 = tmp2 + tmp3
    tmp5 = tl.math.floor(tmp4)
    tmp6 = 1.0
    tmp7 = tmp5 - tmp6
    tmp8 = 0.0
    tmp9 = tmp7 >= tmp8
    tmp10 = 456.0
    tmp11 = tmp7 < tmp10
    tmp13 = 172.5
    tmp14 = tmp12 * tmp13
    tmp15 = 172.0
    tmp16 = tmp14 + tmp15
    tmp17 = tl.math.floor(tmp16)
    tmp18 = -1.0
    tmp19 = tmp17 + tmp18
    tmp20 = tmp19 >= tmp8
    tmp21 = 345.0
    tmp22 = tmp19 < tmp21
    tmp23 = tmp20 & tmp22
    tmp24 = tmp11 & tmp23
    tmp25 = tmp9 & tmp24
    tmp26 = tmp19.to(tl.int64)
    tmp27 = tl.full([1], 0, tl.int64)
    tmp28 = tl.where(tmp25, tmp26, tmp27)
    tmp29 = tmp7.to(tl.int64)
    tmp30 = tl.where(tmp25, tmp29, tmp27)
    tl.device_assert((0 <= tmp28) & (tmp28 < 345), "index out of bounds: 0 <= tmp28 < 345")
    tl.device_assert((0 <= tmp30) & (tmp30 < 456), "index out of bounds: 0 <= tmp30 < 456")
    tmp31 = tl.load(in_ptr1 + (x1 + (3*tmp30) + (1368*tmp28) + (471960*x2)), xmask)
    tmp32 = tl.full([1], 1, tl.int64)
    tmp33 = tl.where(tmp25, tmp32, tmp27)
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp31 * tmp34
    tmp36 = tmp5 + tmp6
    tmp37 = tmp36 >= tmp8
    tmp38 = tmp36 < tmp10
    tmp39 = tmp38 & tmp23
    tmp40 = tmp37 & tmp39
    tmp41 = tl.where(tmp40, tmp26, tmp27)
    tmp42 = tmp36.to(tl.int64)
    tmp43 = tl.where(tmp40, tmp42, tmp27)
    tl.device_assert((0 <= tmp41) & (tmp41 < 345), "index out of bounds: 0 <= tmp41 < 345")
    tl.device_assert((0 <= tmp43) & (tmp43 < 456), "index out of bounds: 0 <= tmp43 < 456")
    tmp44 = tl.load(in_ptr1 + (x1 + (3*tmp43) + (1368*tmp41) + (471960*x2)), xmask)
    tmp45 = tl.where(tmp40, tmp32, tmp27)
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp44 * tmp46
    tmp48 = 2.0
    tmp49 = tmp5 + tmp48
    tmp50 = tmp49 >= tmp8
    tmp51 = tmp49 < tmp10
    tmp52 = tmp51 & tmp23
    tmp53 = tmp50 & tmp52
    tmp54 = tl.where(tmp53, tmp26, tmp27)
    tmp55 = tmp49.to(tl.int64)
    tmp56 = tl.where(tmp53, tmp55, tmp27)
    tl.device_assert((0 <= tmp54) & (tmp54 < 345), "index out of bounds: 0 <= tmp54 < 345")
    tl.device_assert((0 <= tmp56) & (tmp56 < 456), "index out of bounds: 0 <= tmp56 < 456")
    tmp57 = tl.load(in_ptr1 + (x1 + (3*tmp56) + (1368*tmp54) + (471960*x2)), xmask)
    tmp58 = tl.where(tmp53, tmp32, tmp27)
    tmp59 = tmp58.to(tl.float32)
    tmp60 = tmp57 * tmp59
    tmp61 = tmp17 + tmp8
    tmp62 = tmp61 >= tmp8
    tmp63 = tmp61 < tmp21
    tmp64 = tmp62 & tmp63
    tmp65 = tmp11 & tmp64
    tmp66 = tmp9 & tmp65
    tmp67 = tmp61.to(tl.int64)
    tmp68 = tl.where(tmp66, tmp67, tmp27)
    tmp69 = tl.where(tmp66, tmp29, tmp27)
    tl.device_assert((0 <= tmp68) & (tmp68 < 345), "index out of bounds: 0 <= tmp68 < 345")
    tl.device_assert((0 <= tmp69) & (tmp69 < 456), "index out of bounds: 0 <= tmp69 < 456")
    tmp70 = tl.load(in_ptr1 + (x1 + (3*tmp69) + (1368*tmp68) + (471960*x2)), xmask)
    tmp71 = tl.where(tmp66, tmp32, tmp27)
    tmp72 = tmp71.to(tl.float32)
    tmp73 = tmp70 * tmp72
    tmp74 = tmp38 & tmp64
    tmp75 = tmp37 & tmp74
    tmp76 = tl.where(tmp75, tmp67, tmp27)
    tmp77 = tl.where(tmp75, tmp42, tmp27)
    tl.device_assert((0 <= tmp76) & (tmp76 < 345), "index out of bounds: 0 <= tmp76 < 345")
    tl.device_assert((0 <= tmp77) & (tmp77 < 456), "index out of bounds: 0 <= tmp77 < 456")
    tmp78 = tl.load(in_ptr1 + (x1 + (3*tmp77) + (1368*tmp76) + (471960*x2)), xmask)
    tmp79 = tl.where(tmp75, tmp32, tmp27)
    tmp80 = tmp79.to(tl.float32)
    tmp81 = tmp78 * tmp80
    tmp82 = tmp51 & tmp64
    tmp83 = tmp50 & tmp82
    tmp84 = tl.where(tmp83, tmp67, tmp27)
    tmp85 = tl.where(tmp83, tmp55, tmp27)
    tl.device_assert((0 <= tmp84) & (tmp84 < 345), "index out of bounds: 0 <= tmp84 < 345")
    tl.device_assert((0 <= tmp85) & (tmp85 < 456), "index out of bounds: 0 <= tmp85 < 456")
    tmp86 = tl.load(in_ptr1 + (x1 + (3*tmp85) + (1368*tmp84) + (471960*x2)), xmask)
    tmp87 = tl.where(tmp83, tmp32, tmp27)
    tmp88 = tmp87.to(tl.float32)
    tmp89 = tmp86 * tmp88
    tmp90 = tmp17 + tmp6
    tmp91 = tmp90 >= tmp8
    tmp92 = tmp90 < tmp21
    tmp93 = tmp91 & tmp92
    tmp94 = tmp11 & tmp93
    tmp95 = tmp9 & tmp94
    tmp96 = tmp90.to(tl.int64)
    tmp97 = tl.where(tmp95, tmp96, tmp27)
    tmp98 = tl.where(tmp95, tmp29, tmp27)
    tl.device_assert((0 <= tmp97) & (tmp97 < 345), "index out of bounds: 0 <= tmp97 < 345")
    tl.device_assert((0 <= tmp98) & (tmp98 < 456), "index out of bounds: 0 <= tmp98 < 456")
    tmp99 = tl.load(in_ptr1 + (x1 + (3*tmp98) + (1368*tmp97) + (471960*x2)), xmask)
    tmp100 = tl.where(tmp95, tmp32, tmp27)
    tmp101 = tmp100.to(tl.float32)
    tmp102 = tmp99 * tmp101
    tmp103 = tmp38 & tmp93
    tmp104 = tmp37 & tmp103
    tmp105 = tl.where(tmp104, tmp96, tmp27)
    tmp106 = tl.where(tmp104, tmp42, tmp27)
    tl.device_assert((0 <= tmp105) & (tmp105 < 345), "index out of bounds: 0 <= tmp105 < 345")
    tl.device_assert((0 <= tmp106) & (tmp106 < 456), "index out of bounds: 0 <= tmp106 < 456")
    tmp107 = tl.load(in_ptr1 + (x1 + (3*tmp106) + (1368*tmp105) + (471960*x2)), xmask)
    tmp108 = tl.where(tmp104, tmp32, tmp27)
    tmp109 = tmp108.to(tl.float32)
    tmp110 = tmp107 * tmp109
    tmp111 = tmp51 & tmp93
    tmp112 = tmp50 & tmp111
    tmp113 = tl.where(tmp112, tmp96, tmp27)
    tmp114 = tl.where(tmp112, tmp55, tmp27)
    tl.device_assert((0 <= tmp113) & (tmp113 < 345), "index out of bounds: 0 <= tmp113 < 345")
    tl.device_assert((0 <= tmp114) & (tmp114 < 456), "index out of bounds: 0 <= tmp114 < 456")
    tmp115 = tl.load(in_ptr1 + (x1 + (3*tmp114) + (1368*tmp113) + (471960*x2)), xmask)
    tmp116 = tl.where(tmp112, tmp32, tmp27)
    tmp117 = tmp116.to(tl.float32)
    tmp118 = tmp115 * tmp117
    tmp119 = tmp17 + tmp48
    tmp120 = tmp119 >= tmp8
    tmp121 = tmp119 < tmp21
    tmp122 = tmp120 & tmp121
    tmp123 = tmp11 & tmp122
    tmp124 = tmp9 & tmp123
    tmp125 = tmp119.to(tl.int64)
    tmp126 = tl.where(tmp124, tmp125, tmp27)
    tmp127 = tl.where(tmp124, tmp29, tmp27)
    tl.device_assert((0 <= tmp126) & (tmp126 < 345), "index out of bounds: 0 <= tmp126 < 345")
    tl.device_assert((0 <= tmp127) & (tmp127 < 456), "index out of bounds: 0 <= tmp127 < 456")
    tmp128 = tl.load(in_ptr1 + (x1 + (3*tmp127) + (1368*tmp126) + (471960*x2)), xmask)
    tmp129 = tl.where(tmp124, tmp32, tmp27)
    tmp130 = tmp129.to(tl.float32)
    tmp131 = tmp128 * tmp130
    tmp132 = tmp38 & tmp122
    tmp133 = tmp37 & tmp132
    tmp134 = tl.where(tmp133, tmp125, tmp27)
    tmp135 = tl.where(tmp133, tmp42, tmp27)
    tl.device_assert((0 <= tmp134) & (tmp134 < 345), "index out of bounds: 0 <= tmp134 < 345")
    tl.device_assert((0 <= tmp135) & (tmp135 < 456), "index out of bounds: 0 <= tmp135 < 456")
    tmp136 = tl.load(in_ptr1 + (x1 + (3*tmp135) + (1368*tmp134) + (471960*x2)), xmask)
    tmp137 = tl.where(tmp133, tmp32, tmp27)
    tmp138 = tmp137.to(tl.float32)
    tmp139 = tmp136 * tmp138
    tmp140 = tmp51 & tmp122
    tmp141 = tmp50 & tmp140
    tmp142 = tl.where(tmp141, tmp125, tmp27)
    tmp143 = tl.where(tmp141, tmp55, tmp27)
    tl.device_assert((0 <= tmp142) & (tmp142 < 345), "index out of bounds: 0 <= tmp142 < 345")
    tl.device_assert((0 <= tmp143) & (tmp143 < 456), "index out of bounds: 0 <= tmp143 < 456")
    tmp144 = tl.load(in_ptr1 + (x1 + (3*tmp143) + (1368*tmp142) + (471960*x2)), xmask)
    tmp145 = tl.where(tmp141, tmp32, tmp27)
    tmp146 = tmp145.to(tl.float32)
    tmp147 = tmp144 * tmp146
    tmp148 = tmp4 - tmp5
    tmp149 = tmp148 + tmp6
    tmp150 = -0.75
    tmp151 = tmp149 * tmp150
    tmp152 = -3.75
    tmp153 = tmp151 - tmp152
    tmp154 = tmp153 * tmp149
    tmp155 = -6.0
    tmp156 = tmp154 + tmp155
    tmp157 = tmp156 * tmp149
    tmp158 = -3.0
    tmp159 = tmp157 - tmp158
    tmp160 = tmp35 * tmp159
    tmp161 = tmp6 - tmp148
    tmp162 = 1.25
    tmp163 = tmp161 * tmp162
    tmp164 = 2.25
    tmp165 = tmp163 - tmp164
    tmp166 = tmp165 * tmp161
    tmp167 = tmp166 * tmp161
    tmp168 = tmp167 + tmp6
    tmp169 = tmp47 * tmp168
    tmp170 = tmp48 - tmp148
    tmp171 = tmp170 * tmp150
    tmp172 = tmp171 - tmp152
    tmp173 = tmp172 * tmp170
    tmp174 = tmp173 + tmp155
    tmp175 = tmp174 * tmp170
    tmp176 = tmp175 - tmp158
    tmp177 = tmp60 * tmp176
    tmp178 = tmp73 * tmp159
    tmp179 = tmp81 * tmp168
    tmp180 = tmp89 * tmp176
    tmp181 = tmp102 * tmp159
    tmp182 = tmp110 * tmp168
    tmp183 = tmp118 * tmp176
    tmp184 = tmp131 * tmp159
    tmp185 = tmp139 * tmp168
    tmp186 = tmp147 * tmp176
    tmp187 = tmp5 >= tmp8
    tmp188 = tmp5 < tmp10
    tmp189 = tmp188 & tmp23
    tmp190 = tmp187 & tmp189
    tmp191 = tl.where(tmp190, tmp26, tmp27)
    tl.device_assert((0 <= tmp191) & (tmp191 < 345), "index out of bounds: 0 <= tmp191 < 345")
    tmp192 = tmp5.to(tl.int64)
    tmp193 = tl.where(tmp190, tmp192, tmp27)
    tl.device_assert((0 <= tmp193) & (tmp193 < 456), "index out of bounds: 0 <= tmp193 < 456")
    tmp194 = tl.load(in_ptr1 + (x1 + (3*tmp193) + (1368*tmp191) + (471960*x2)), xmask)
    tmp195 = tl.where(tmp190, tmp32, tmp27)
    tmp196 = tmp195.to(tl.float32)
    tmp197 = tmp194 * tmp196
    tmp198 = tmp148 * tmp162
    tmp199 = tmp198 - tmp164
    tmp200 = tmp199 * tmp148
    tmp201 = tmp200 * tmp148
    tmp202 = tmp201 + tmp6
    tmp203 = tmp197 * tmp202
    tmp204 = tmp188 & tmp64
    tmp205 = tmp187 & tmp204
    tmp206 = tl.where(tmp205, tmp67, tmp27)
    tl.device_assert((0 <= tmp206) & (tmp206 < 345), "index out of bounds: 0 <= tmp206 < 345")
    tmp207 = tl.where(tmp205, tmp192, tmp27)
    tl.device_assert((0 <= tmp207) & (tmp207 < 456), "index out of bounds: 0 <= tmp207 < 456")
    tmp208 = tl.load(in_ptr1 + (x1 + (3*tmp207) + (1368*tmp206) + (471960*x2)), xmask)
    tmp209 = tl.where(tmp205, tmp32, tmp27)
    tmp210 = tmp209.to(tl.float32)
    tmp211 = tmp208 * tmp210
    tmp212 = tmp211 * tmp202
    tmp213 = tmp188 & tmp93
    tmp214 = tmp187 & tmp213
    tmp215 = tl.where(tmp214, tmp96, tmp27)
    tl.device_assert((0 <= tmp215) & (tmp215 < 345), "index out of bounds: 0 <= tmp215 < 345")
    tmp216 = tl.where(tmp214, tmp192, tmp27)
    tl.device_assert((0 <= tmp216) & (tmp216 < 456), "index out of bounds: 0 <= tmp216 < 456")
    tmp217 = tl.load(in_ptr1 + (x1 + (3*tmp216) + (1368*tmp215) + (471960*x2)), xmask)
    tmp218 = tl.where(tmp214, tmp32, tmp27)
    tmp219 = tmp218.to(tl.float32)
    tmp220 = tmp217 * tmp219
    tmp221 = tmp220 * tmp202
    tmp222 = tmp188 & tmp122
    tmp223 = tmp187 & tmp222
    tmp224 = tl.where(tmp223, tmp125, tmp27)
    tl.device_assert((0 <= tmp224) & (tmp224 < 345), "index out of bounds: 0 <= tmp224 < 345")
    tmp225 = tl.where(tmp223, tmp192, tmp27)
    tl.device_assert((0 <= tmp225) & (tmp225 < 456), "index out of bounds: 0 <= tmp225 < 456")
    tmp226 = tl.load(in_ptr1 + (x1 + (3*tmp225) + (1368*tmp224) + (471960*x2)), xmask)
    tmp227 = tl.where(tmp223, tmp32, tmp27)
    tmp228 = tmp227.to(tl.float32)
    tmp229 = tmp226 * tmp228
    tmp230 = tmp229 * tmp202
    tmp231 = tmp160 + tmp203
    tmp232 = tmp231 + tmp169
    tmp233 = tmp232 + tmp177
    tmp234 = tmp16 - tmp17
    tmp235 = tmp234 + tmp6
    tmp236 = tmp235 * tmp150
    tmp237 = tmp236 - tmp152
    tmp238 = tmp237 * tmp235
    tmp239 = tmp238 + tmp155
    tmp240 = tmp239 * tmp235
    tmp241 = tmp240 - tmp158
    tmp242 = tmp233 * tmp241
    tmp243 = tmp178 + tmp212
    tmp244 = tmp243 + tmp179
    tmp245 = tmp244 + tmp180
    tmp246 = tmp234 * tmp162
    tmp247 = tmp246 - tmp164
    tmp248 = tmp247 * tmp234
    tmp249 = tmp248 * tmp234
    tmp250 = tmp249 + tmp6
    tmp251 = tmp245 * tmp250
    tmp252 = tmp181 + tmp221
    tmp253 = tmp252 + tmp182
    tmp254 = tmp253 + tmp183
    tmp255 = tmp6 - tmp234
    tmp256 = tmp255 * tmp162
    tmp257 = tmp256 - tmp164
    tmp258 = tmp257 * tmp255
    tmp259 = tmp258 * tmp255
    tmp260 = tmp259 + tmp6
    tmp261 = tmp254 * tmp260
    tmp262 = tmp184 + tmp230
    tmp263 = tmp262 + tmp185
    tmp264 = tmp263 + tmp186
    tmp265 = tmp48 - tmp234
    tmp266 = tmp265 * tmp150
    tmp267 = tmp266 - tmp152
    tmp268 = tmp267 * tmp265
    tmp269 = tmp268 + tmp155
    tmp270 = tmp269 * tmp265
    tmp271 = tmp270 - tmp158
    tmp272 = tmp264 * tmp271
    tmp273 = tmp242 + tmp251
    tmp274 = tmp273 + tmp261
    tmp275 = tmp274 + tmp272
    tl.store(in_out_ptr1 + (x3), tmp275, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (2, 3, 345, 456), (471960, 1, 1368, 3))
    assert_size_stride(arg1_1, (2, 2, 3), (6, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf1 = empty_strided((2, 157320, 2), (314640, 2, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [affine_grid], Original ATen: [aten.affine_grid_generator]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_affine_grid_generator_0.run(arg1_1, buf1, 629280, grid=grid(629280), stream=stream0)
        del arg1_1
        buf10 = empty_strided((2, 3, 345, 456), (471960, 157320, 456, 1), device='cuda', dtype=torch.float32)
        buf11 = buf10; del buf10  # reuse
        buf16 = buf11; del buf11  # reuse
        buf62 = buf16; del buf16  # reuse
        # Source Nodes: [grid_sample], Original ATen: [aten.grid_sampler_2d]
        triton_poi_fused_grid_sampler_2d_1.run(buf62, buf1, arg0_1, 943920, grid=grid(943920), stream=stream0)
        del arg0_1
        return (buf62, )


def benchmark_compiled_module(times=1000, repeat=20):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 3, 345, 456), (471960, 1, 1368, 3), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((2, 2, 3), (6, 3, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
