
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


# kernel path: /tmp/torchinductor_root/6b/c6bshgcki4kdacgepg5rndcv25vaf4j2i57stz3r7gfryvwgmws3.py
# Original ATen: aten.affine_grid_generator

# aten.affine_grid_generator => mul_4, sum_1
triton_poi_fused_affine_grid_generator_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
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


# kernel path: /tmp/torchinductor_root/gg/cggdy4c3jminl6cabrqdr5oowgg4fe5poh54re6l7dfbf2bgh5x2.py
# Original ATen: aten.grid_sampler_2d

# aten.grid_sampler_2d => add_10, add_11, add_12, add_13, add_14, add_15, add_16, add_17, add_18, add_19, add_20, add_21, add_22, add_23, add_24, add_25, add_26, add_27, add_28, add_29, add_30, add_31, add_32, add_33, add_34, add_35, add_36, add_37, add_38, add_39, add_4, add_40, add_41, add_42, add_43, add_44, add_45, add_46, add_47, add_48, add_49, add_5, add_50, add_51, add_52, add_53, add_54, add_55, add_56, add_57, add_6, add_7, add_8, add_9, convert_element_type_10, convert_element_type_11, convert_element_type_12, convert_element_type_13, convert_element_type_16, convert_element_type_17, convert_element_type_18, convert_element_type_19, convert_element_type_20, convert_element_type_21, convert_element_type_24, convert_element_type_25, convert_element_type_26, convert_element_type_27, convert_element_type_28, convert_element_type_29, convert_element_type_32, convert_element_type_33, convert_element_type_34, convert_element_type_35, convert_element_type_4, convert_element_type_5, convert_element_type_8, convert_element_type_9, floor, floor_1, full_default_1, full_default_10, full_default_11, full_default_12, full_default_13, full_default_14, full_default_15, full_default_16, full_default_17, full_default_18, full_default_19, full_default_2, full_default_20, full_default_23, full_default_24, full_default_25, full_default_26, full_default_27, full_default_28, full_default_29, full_default_3, full_default_30, full_default_31, full_default_32, full_default_33, full_default_34, full_default_35, full_default_36, full_default_39, full_default_4, full_default_40, full_default_41, full_default_42, full_default_43, full_default_44, full_default_45, full_default_46, full_default_47, full_default_48, full_default_49, full_default_50, full_default_51, full_default_52, full_default_55, full_default_56, full_default_57, full_default_58, full_default_59, full_default_60, full_default_61, full_default_62, full_default_63, full_default_64, full_default_7, full_default_8, full_default_9, ge, ge_1, ge_10, ge_11, ge_12, ge_13, ge_14, ge_15, ge_16, ge_17, ge_18, ge_19, ge_2, ge_20, ge_21, ge_22, ge_23, ge_24, ge_25, ge_26, ge_27, ge_28, ge_29, ge_3, ge_30, ge_31, ge_4, ge_5, ge_6, ge_7, ge_8, ge_9, index, index_1, index_10, index_11, index_12, index_13, index_14, index_15, index_2, index_3, index_4, index_5, index_6, index_7, index_8, index_9, logical_and, logical_and_1, logical_and_10, logical_and_11, logical_and_12, logical_and_13, logical_and_14, logical_and_15, logical_and_16, logical_and_17, logical_and_18, logical_and_19, logical_and_2, logical_and_20, logical_and_21, logical_and_22, logical_and_23, logical_and_24, logical_and_25, logical_and_26, logical_and_27, logical_and_28, logical_and_29, logical_and_3, logical_and_30, logical_and_31, logical_and_32, logical_and_33, logical_and_34, logical_and_35, logical_and_36, logical_and_37, logical_and_38, logical_and_39, logical_and_4, logical_and_40, logical_and_41, logical_and_42, logical_and_43, logical_and_44, logical_and_45, logical_and_46, logical_and_47, logical_and_5, logical_and_6, logical_and_7, logical_and_8, logical_and_9, lt_10, lt_11, lt_12, lt_13, lt_14, lt_15, lt_16, lt_17, lt_18, lt_19, lt_2, lt_20, lt_21, lt_22, lt_23, lt_24, lt_25, lt_26, lt_27, lt_28, lt_29, lt_3, lt_30, lt_31, lt_32, lt_33, lt_4, lt_5, lt_6, lt_7, lt_8, lt_9, mul_10, mul_100, mul_101, mul_102, mul_11, mul_12, mul_13, mul_14, mul_15, mul_16, mul_17, mul_18, mul_19, mul_20, mul_21, mul_22, mul_23, mul_24, mul_25, mul_26, mul_27, mul_28, mul_29, mul_30, mul_31, mul_32, mul_33, mul_34, mul_35, mul_36, mul_37, mul_38, mul_39, mul_40, mul_41, mul_42, mul_43, mul_44, mul_45, mul_46, mul_47, mul_48, mul_49, mul_5, mul_50, mul_51, mul_52, mul_53, mul_54, mul_55, mul_56, mul_57, mul_58, mul_59, mul_6, mul_60, mul_61, mul_62, mul_63, mul_64, mul_65, mul_66, mul_67, mul_68, mul_69, mul_7, mul_70, mul_71, mul_72, mul_73, mul_74, mul_75, mul_76, mul_77, mul_78, mul_79, mul_8, mul_80, mul_81, mul_82, mul_83, mul_84, mul_85, mul_86, mul_87, mul_88, mul_89, mul_9, mul_90, mul_91, mul_92, mul_93, mul_94, mul_95, mul_96, mul_97, mul_98, mul_99, sub_10, sub_11, sub_12, sub_13, sub_14, sub_15, sub_16, sub_17, sub_18, sub_19, sub_20, sub_21, sub_22, sub_23, sub_24, sub_25, sub_26, sub_27, sub_28, sub_29, sub_30, sub_31, sub_32, sub_33, sub_34, sub_35, sub_36, sub_37, sub_38, sub_39, sub_4, sub_40, sub_41, sub_42, sub_43, sub_44, sub_45, sub_46, sub_47, sub_48, sub_49, sub_5, sub_6, sub_7, sub_8, sub_9, where_10, where_11, where_12, where_13, where_14, where_15, where_16, where_19, where_2, where_20, where_21, where_22, where_23, where_24, where_25, where_26, where_27, where_28, where_3, where_31, where_32, where_33, where_34, where_35, where_36, where_37, where_38, where_39, where_4, where_40, where_43, where_44, where_45, where_46, where_47, where_48, where_49, where_7, where_8, where_9
triton_poi_fused_grid_sampler_2d_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'constants': {}, 'mutated_arg_names': ['in_out_ptr1'], 'autotune_hints': set(), 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=())]})
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
    tmp31 = triton_helpers.promote_to_tensor(tmp28)
    tl.device_assert((0 <= tmp31) & (tmp31 < 345), "index out of bounds: 0 <= tmp31 < 345")
    tmp32 = triton_helpers.promote_to_tensor(tmp30)
    tl.device_assert((0 <= tmp32) & (tmp32 < 456), "index out of bounds: 0 <= tmp32 < 456")
    tmp33 = tl.load(in_ptr1 + (x1 + (3*tmp30) + (1368*tmp28) + (471960*x2)), xmask)
    tmp34 = tl.full([1], 1, tl.int64)
    tmp35 = tl.where(tmp25, tmp34, tmp27)
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp33 * tmp36
    tmp38 = tmp5 + tmp6
    tmp39 = tmp38 >= tmp8
    tmp40 = tmp38 < tmp10
    tmp41 = tmp40 & tmp23
    tmp42 = tmp39 & tmp41
    tmp43 = tl.where(tmp42, tmp26, tmp27)
    tmp44 = tmp38.to(tl.int64)
    tmp45 = tl.where(tmp42, tmp44, tmp27)
    tmp46 = triton_helpers.promote_to_tensor(tmp43)
    tl.device_assert((0 <= tmp46) & (tmp46 < 345), "index out of bounds: 0 <= tmp46 < 345")
    tmp47 = triton_helpers.promote_to_tensor(tmp45)
    tl.device_assert((0 <= tmp47) & (tmp47 < 456), "index out of bounds: 0 <= tmp47 < 456")
    tmp48 = tl.load(in_ptr1 + (x1 + (3*tmp45) + (1368*tmp43) + (471960*x2)), xmask)
    tmp49 = tl.where(tmp42, tmp34, tmp27)
    tmp50 = tmp49.to(tl.float32)
    tmp51 = tmp48 * tmp50
    tmp52 = 2.0
    tmp53 = tmp5 + tmp52
    tmp54 = tmp53 >= tmp8
    tmp55 = tmp53 < tmp10
    tmp56 = tmp55 & tmp23
    tmp57 = tmp54 & tmp56
    tmp58 = tl.where(tmp57, tmp26, tmp27)
    tmp59 = tmp53.to(tl.int64)
    tmp60 = tl.where(tmp57, tmp59, tmp27)
    tmp61 = triton_helpers.promote_to_tensor(tmp58)
    tl.device_assert((0 <= tmp61) & (tmp61 < 345), "index out of bounds: 0 <= tmp61 < 345")
    tmp62 = triton_helpers.promote_to_tensor(tmp60)
    tl.device_assert((0 <= tmp62) & (tmp62 < 456), "index out of bounds: 0 <= tmp62 < 456")
    tmp63 = tl.load(in_ptr1 + (x1 + (3*tmp60) + (1368*tmp58) + (471960*x2)), xmask)
    tmp64 = tl.where(tmp57, tmp34, tmp27)
    tmp65 = tmp64.to(tl.float32)
    tmp66 = tmp63 * tmp65
    tmp67 = tmp17 + tmp8
    tmp68 = tmp67 >= tmp8
    tmp69 = tmp67 < tmp21
    tmp70 = tmp68 & tmp69
    tmp71 = tmp11 & tmp70
    tmp72 = tmp9 & tmp71
    tmp73 = tmp67.to(tl.int64)
    tmp74 = tl.where(tmp72, tmp73, tmp27)
    tmp75 = tl.where(tmp72, tmp29, tmp27)
    tmp76 = triton_helpers.promote_to_tensor(tmp74)
    tl.device_assert((0 <= tmp76) & (tmp76 < 345), "index out of bounds: 0 <= tmp76 < 345")
    tmp77 = triton_helpers.promote_to_tensor(tmp75)
    tl.device_assert((0 <= tmp77) & (tmp77 < 456), "index out of bounds: 0 <= tmp77 < 456")
    tmp78 = tl.load(in_ptr1 + (x1 + (3*tmp75) + (1368*tmp74) + (471960*x2)), xmask)
    tmp79 = tl.where(tmp72, tmp34, tmp27)
    tmp80 = tmp79.to(tl.float32)
    tmp81 = tmp78 * tmp80
    tmp82 = tmp40 & tmp70
    tmp83 = tmp39 & tmp82
    tmp84 = tl.where(tmp83, tmp73, tmp27)
    tmp85 = tl.where(tmp83, tmp44, tmp27)
    tmp86 = triton_helpers.promote_to_tensor(tmp84)
    tl.device_assert((0 <= tmp86) & (tmp86 < 345), "index out of bounds: 0 <= tmp86 < 345")
    tmp87 = triton_helpers.promote_to_tensor(tmp85)
    tl.device_assert((0 <= tmp87) & (tmp87 < 456), "index out of bounds: 0 <= tmp87 < 456")
    tmp88 = tl.load(in_ptr1 + (x1 + (3*tmp85) + (1368*tmp84) + (471960*x2)), xmask)
    tmp89 = tl.where(tmp83, tmp34, tmp27)
    tmp90 = tmp89.to(tl.float32)
    tmp91 = tmp88 * tmp90
    tmp92 = tmp55 & tmp70
    tmp93 = tmp54 & tmp92
    tmp94 = tl.where(tmp93, tmp73, tmp27)
    tmp95 = tl.where(tmp93, tmp59, tmp27)
    tmp96 = triton_helpers.promote_to_tensor(tmp94)
    tl.device_assert((0 <= tmp96) & (tmp96 < 345), "index out of bounds: 0 <= tmp96 < 345")
    tmp97 = triton_helpers.promote_to_tensor(tmp95)
    tl.device_assert((0 <= tmp97) & (tmp97 < 456), "index out of bounds: 0 <= tmp97 < 456")
    tmp98 = tl.load(in_ptr1 + (x1 + (3*tmp95) + (1368*tmp94) + (471960*x2)), xmask)
    tmp99 = tl.where(tmp93, tmp34, tmp27)
    tmp100 = tmp99.to(tl.float32)
    tmp101 = tmp98 * tmp100
    tmp102 = tmp17 + tmp6
    tmp103 = tmp102 >= tmp8
    tmp104 = tmp102 < tmp21
    tmp105 = tmp103 & tmp104
    tmp106 = tmp11 & tmp105
    tmp107 = tmp9 & tmp106
    tmp108 = tmp102.to(tl.int64)
    tmp109 = tl.where(tmp107, tmp108, tmp27)
    tmp110 = tl.where(tmp107, tmp29, tmp27)
    tmp111 = triton_helpers.promote_to_tensor(tmp109)
    tl.device_assert((0 <= tmp111) & (tmp111 < 345), "index out of bounds: 0 <= tmp111 < 345")
    tmp112 = triton_helpers.promote_to_tensor(tmp110)
    tl.device_assert((0 <= tmp112) & (tmp112 < 456), "index out of bounds: 0 <= tmp112 < 456")
    tmp113 = tl.load(in_ptr1 + (x1 + (3*tmp110) + (1368*tmp109) + (471960*x2)), xmask)
    tmp114 = tl.where(tmp107, tmp34, tmp27)
    tmp115 = tmp114.to(tl.float32)
    tmp116 = tmp113 * tmp115
    tmp117 = tmp40 & tmp105
    tmp118 = tmp39 & tmp117
    tmp119 = tl.where(tmp118, tmp108, tmp27)
    tmp120 = tl.where(tmp118, tmp44, tmp27)
    tmp121 = triton_helpers.promote_to_tensor(tmp119)
    tl.device_assert((0 <= tmp121) & (tmp121 < 345), "index out of bounds: 0 <= tmp121 < 345")
    tmp122 = triton_helpers.promote_to_tensor(tmp120)
    tl.device_assert((0 <= tmp122) & (tmp122 < 456), "index out of bounds: 0 <= tmp122 < 456")
    tmp123 = tl.load(in_ptr1 + (x1 + (3*tmp120) + (1368*tmp119) + (471960*x2)), xmask)
    tmp124 = tl.where(tmp118, tmp34, tmp27)
    tmp125 = tmp124.to(tl.float32)
    tmp126 = tmp123 * tmp125
    tmp127 = tmp55 & tmp105
    tmp128 = tmp54 & tmp127
    tmp129 = tl.where(tmp128, tmp108, tmp27)
    tmp130 = tl.where(tmp128, tmp59, tmp27)
    tmp131 = triton_helpers.promote_to_tensor(tmp129)
    tl.device_assert((0 <= tmp131) & (tmp131 < 345), "index out of bounds: 0 <= tmp131 < 345")
    tmp132 = triton_helpers.promote_to_tensor(tmp130)
    tl.device_assert((0 <= tmp132) & (tmp132 < 456), "index out of bounds: 0 <= tmp132 < 456")
    tmp133 = tl.load(in_ptr1 + (x1 + (3*tmp130) + (1368*tmp129) + (471960*x2)), xmask)
    tmp134 = tl.where(tmp128, tmp34, tmp27)
    tmp135 = tmp134.to(tl.float32)
    tmp136 = tmp133 * tmp135
    tmp137 = tmp17 + tmp52
    tmp138 = tmp137 >= tmp8
    tmp139 = tmp137 < tmp21
    tmp140 = tmp138 & tmp139
    tmp141 = tmp11 & tmp140
    tmp142 = tmp9 & tmp141
    tmp143 = tmp137.to(tl.int64)
    tmp144 = tl.where(tmp142, tmp143, tmp27)
    tmp145 = tl.where(tmp142, tmp29, tmp27)
    tmp146 = triton_helpers.promote_to_tensor(tmp144)
    tl.device_assert((0 <= tmp146) & (tmp146 < 345), "index out of bounds: 0 <= tmp146 < 345")
    tmp147 = triton_helpers.promote_to_tensor(tmp145)
    tl.device_assert((0 <= tmp147) & (tmp147 < 456), "index out of bounds: 0 <= tmp147 < 456")
    tmp148 = tl.load(in_ptr1 + (x1 + (3*tmp145) + (1368*tmp144) + (471960*x2)), xmask)
    tmp149 = tl.where(tmp142, tmp34, tmp27)
    tmp150 = tmp149.to(tl.float32)
    tmp151 = tmp148 * tmp150
    tmp152 = tmp40 & tmp140
    tmp153 = tmp39 & tmp152
    tmp154 = tl.where(tmp153, tmp143, tmp27)
    tmp155 = tl.where(tmp153, tmp44, tmp27)
    tmp156 = triton_helpers.promote_to_tensor(tmp154)
    tl.device_assert((0 <= tmp156) & (tmp156 < 345), "index out of bounds: 0 <= tmp156 < 345")
    tmp157 = triton_helpers.promote_to_tensor(tmp155)
    tl.device_assert((0 <= tmp157) & (tmp157 < 456), "index out of bounds: 0 <= tmp157 < 456")
    tmp158 = tl.load(in_ptr1 + (x1 + (3*tmp155) + (1368*tmp154) + (471960*x2)), xmask)
    tmp159 = tl.where(tmp153, tmp34, tmp27)
    tmp160 = tmp159.to(tl.float32)
    tmp161 = tmp158 * tmp160
    tmp162 = tmp55 & tmp140
    tmp163 = tmp54 & tmp162
    tmp164 = tl.where(tmp163, tmp143, tmp27)
    tmp165 = tl.where(tmp163, tmp59, tmp27)
    tmp166 = triton_helpers.promote_to_tensor(tmp164)
    tl.device_assert((0 <= tmp166) & (tmp166 < 345), "index out of bounds: 0 <= tmp166 < 345")
    tmp167 = triton_helpers.promote_to_tensor(tmp165)
    tl.device_assert((0 <= tmp167) & (tmp167 < 456), "index out of bounds: 0 <= tmp167 < 456")
    tmp168 = tl.load(in_ptr1 + (x1 + (3*tmp165) + (1368*tmp164) + (471960*x2)), xmask)
    tmp169 = tl.where(tmp163, tmp34, tmp27)
    tmp170 = tmp169.to(tl.float32)
    tmp171 = tmp168 * tmp170
    tmp172 = tmp4 - tmp5
    tmp173 = tmp172 + tmp6
    tmp174 = -0.75
    tmp175 = tmp173 * tmp174
    tmp176 = -3.75
    tmp177 = tmp175 - tmp176
    tmp178 = tmp177 * tmp173
    tmp179 = -6.0
    tmp180 = tmp178 + tmp179
    tmp181 = tmp180 * tmp173
    tmp182 = -3.0
    tmp183 = tmp181 - tmp182
    tmp184 = tmp37 * tmp183
    tmp185 = tmp6 - tmp172
    tmp186 = 1.25
    tmp187 = tmp185 * tmp186
    tmp188 = 2.25
    tmp189 = tmp187 - tmp188
    tmp190 = tmp189 * tmp185
    tmp191 = tmp190 * tmp185
    tmp192 = tmp191 + tmp6
    tmp193 = tmp51 * tmp192
    tmp194 = tmp52 - tmp172
    tmp195 = tmp194 * tmp174
    tmp196 = tmp195 - tmp176
    tmp197 = tmp196 * tmp194
    tmp198 = tmp197 + tmp179
    tmp199 = tmp198 * tmp194
    tmp200 = tmp199 - tmp182
    tmp201 = tmp66 * tmp200
    tmp202 = tmp81 * tmp183
    tmp203 = tmp91 * tmp192
    tmp204 = tmp101 * tmp200
    tmp205 = tmp116 * tmp183
    tmp206 = tmp126 * tmp192
    tmp207 = tmp136 * tmp200
    tmp208 = tmp151 * tmp183
    tmp209 = tmp161 * tmp192
    tmp210 = tmp171 * tmp200
    tmp211 = tmp5 >= tmp8
    tmp212 = tmp5 < tmp10
    tmp213 = tmp212 & tmp23
    tmp214 = tmp211 & tmp213
    tmp215 = tl.where(tmp214, tmp26, tmp27)
    tmp216 = triton_helpers.promote_to_tensor(tmp215)
    tl.device_assert((0 <= tmp216) & (tmp216 < 345), "index out of bounds: 0 <= tmp216 < 345")
    tmp217 = tmp5.to(tl.int64)
    tmp218 = tl.where(tmp214, tmp217, tmp27)
    tmp219 = triton_helpers.promote_to_tensor(tmp218)
    tl.device_assert((0 <= tmp219) & (tmp219 < 456), "index out of bounds: 0 <= tmp219 < 456")
    tmp220 = tl.load(in_ptr1 + (x1 + (3*tmp218) + (1368*tmp215) + (471960*x2)), xmask)
    tmp221 = tl.where(tmp214, tmp34, tmp27)
    tmp222 = tmp221.to(tl.float32)
    tmp223 = tmp220 * tmp222
    tmp224 = tmp172 * tmp186
    tmp225 = tmp224 - tmp188
    tmp226 = tmp225 * tmp172
    tmp227 = tmp226 * tmp172
    tmp228 = tmp227 + tmp6
    tmp229 = tmp223 * tmp228
    tmp230 = tmp212 & tmp70
    tmp231 = tmp211 & tmp230
    tmp232 = tl.where(tmp231, tmp73, tmp27)
    tmp233 = triton_helpers.promote_to_tensor(tmp232)
    tl.device_assert((0 <= tmp233) & (tmp233 < 345), "index out of bounds: 0 <= tmp233 < 345")
    tmp234 = tl.where(tmp231, tmp217, tmp27)
    tmp235 = triton_helpers.promote_to_tensor(tmp234)
    tl.device_assert((0 <= tmp235) & (tmp235 < 456), "index out of bounds: 0 <= tmp235 < 456")
    tmp236 = tl.load(in_ptr1 + (x1 + (3*tmp234) + (1368*tmp232) + (471960*x2)), xmask)
    tmp237 = tl.where(tmp231, tmp34, tmp27)
    tmp238 = tmp237.to(tl.float32)
    tmp239 = tmp236 * tmp238
    tmp240 = tmp239 * tmp228
    tmp241 = tmp212 & tmp105
    tmp242 = tmp211 & tmp241
    tmp243 = tl.where(tmp242, tmp108, tmp27)
    tmp244 = triton_helpers.promote_to_tensor(tmp243)
    tl.device_assert((0 <= tmp244) & (tmp244 < 345), "index out of bounds: 0 <= tmp244 < 345")
    tmp245 = tl.where(tmp242, tmp217, tmp27)
    tmp246 = triton_helpers.promote_to_tensor(tmp245)
    tl.device_assert((0 <= tmp246) & (tmp246 < 456), "index out of bounds: 0 <= tmp246 < 456")
    tmp247 = tl.load(in_ptr1 + (x1 + (3*tmp245) + (1368*tmp243) + (471960*x2)), xmask)
    tmp248 = tl.where(tmp242, tmp34, tmp27)
    tmp249 = tmp248.to(tl.float32)
    tmp250 = tmp247 * tmp249
    tmp251 = tmp250 * tmp228
    tmp252 = tmp212 & tmp140
    tmp253 = tmp211 & tmp252
    tmp254 = tl.where(tmp253, tmp143, tmp27)
    tmp255 = triton_helpers.promote_to_tensor(tmp254)
    tl.device_assert((0 <= tmp255) & (tmp255 < 345), "index out of bounds: 0 <= tmp255 < 345")
    tmp256 = tl.where(tmp253, tmp217, tmp27)
    tmp257 = triton_helpers.promote_to_tensor(tmp256)
    tl.device_assert((0 <= tmp257) & (tmp257 < 456), "index out of bounds: 0 <= tmp257 < 456")
    tmp258 = tl.load(in_ptr1 + (x1 + (3*tmp256) + (1368*tmp254) + (471960*x2)), xmask)
    tmp259 = tl.where(tmp253, tmp34, tmp27)
    tmp260 = tmp259.to(tl.float32)
    tmp261 = tmp258 * tmp260
    tmp262 = tmp261 * tmp228
    tmp263 = tmp184 + tmp229
    tmp264 = tmp263 + tmp193
    tmp265 = tmp264 + tmp201
    tmp266 = tmp16 - tmp17
    tmp267 = tmp266 + tmp6
    tmp268 = tmp267 * tmp174
    tmp269 = tmp268 - tmp176
    tmp270 = tmp269 * tmp267
    tmp271 = tmp270 + tmp179
    tmp272 = tmp271 * tmp267
    tmp273 = tmp272 - tmp182
    tmp274 = tmp265 * tmp273
    tmp275 = tmp202 + tmp240
    tmp276 = tmp275 + tmp203
    tmp277 = tmp276 + tmp204
    tmp278 = tmp266 * tmp186
    tmp279 = tmp278 - tmp188
    tmp280 = tmp279 * tmp266
    tmp281 = tmp280 * tmp266
    tmp282 = tmp281 + tmp6
    tmp283 = tmp277 * tmp282
    tmp284 = tmp205 + tmp251
    tmp285 = tmp284 + tmp206
    tmp286 = tmp285 + tmp207
    tmp287 = tmp6 - tmp266
    tmp288 = tmp287 * tmp186
    tmp289 = tmp288 - tmp188
    tmp290 = tmp289 * tmp287
    tmp291 = tmp290 * tmp287
    tmp292 = tmp291 + tmp6
    tmp293 = tmp286 * tmp292
    tmp294 = tmp208 + tmp262
    tmp295 = tmp294 + tmp209
    tmp296 = tmp295 + tmp210
    tmp297 = tmp52 - tmp266
    tmp298 = tmp297 * tmp174
    tmp299 = tmp298 - tmp176
    tmp300 = tmp299 * tmp297
    tmp301 = tmp300 + tmp179
    tmp302 = tmp301 * tmp297
    tmp303 = tmp302 - tmp182
    tmp304 = tmp296 * tmp303
    tmp305 = tmp274 + tmp283
    tmp306 = tmp305 + tmp293
    tmp307 = tmp306 + tmp304
    tl.store(in_out_ptr1 + (x3), tmp307, xmask)
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
        stream0 = get_cuda_stream(0)
        triton_poi_fused_affine_grid_generator_0.run(arg1_1, buf1, 629280, grid=grid(629280), stream=stream0)
        del arg1_1
        buf10 = empty_strided((2, 3, 345, 456), (471960, 157320, 456, 1), device='cuda', dtype=torch.float32)
        buf11 = buf10; del buf10  # reuse
        buf16 = buf11; del buf11  # reuse
        buf62 = buf16; del buf16  # reuse
        triton_poi_fused_grid_sampler_2d_1.run(buf62, buf1, arg0_1, 943920, grid=grid(943920), stream=stream0)
        del arg0_1
        return (buf62, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 3, 345, 456), (471960, 1, 1368, 3), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((2, 2, 3), (6, 3, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1]), times=times, repeat=repeat)


# if __name__ == "__main__":
#     from torch._inductor.utils import compiled_module_main
#     compiled_module_main('None', benchmark_compiled_module)

from torch.nn.functional import grid_sample, affine_grid


def transform(img, theta):
    n, c, h, w = img.shape
    grid = affine_grid(theta, size=(n, c, h, w), align_corners=False)
    output = grid_sample(img, grid, align_corners=False, mode="bicubic")
    return output


if __name__ == "__main__":

    torch.manual_seed(12)

    memory_format = torch.channels_last
    dtype = torch.float32
    device = "cuda"

    a = torch.deg2rad(torch.tensor(45.0))
    ca, sa = torch.cos(a), torch.sin(a)
    s1 = 1.23
    s2 = 1.34

    n, c, h, w = 2, 3, 345, 456

    theta = torch.tensor([[
        [ca / s1, sa, 0.0],
        [-sa, ca / s2, 0.0],
    ]])
    theta = theta.expand(n, 2, 3).contiguous()

    x = torch.arange(n * c * h * w, device=device).reshape(n, c, h, w).to(torch.uint8)
    x = x.to(dtype=dtype)
    x = x.contiguous(memory_format=memory_format)
    theta = theta.to(device=device, dtype=dtype)

    import torch.utils.benchmark as benchmark

    min_run_time = 10
    results = []
    results.append(
        benchmark.Timer(
            stmt=f"fn(x, m)",
            globals={
                "fn": transform,
                "x": x,
                "m": theta,
            },
            num_threads=torch.get_num_threads(),
            label=f"Affine Grid Sampler 2d",
            sub_label=f"bicubic f32, CL, BS={n}",
            description=f"Eager (Torch {torch.__version__})",
        ).blocked_autorange(min_run_time=min_run_time)
    )
    results.append(
        benchmark.Timer(
            stmt=f"fn([x, m])",
            globals={
                "fn": call,
                "x": x,
                "m": theta,
            },
            num_threads=torch.get_num_threads(),
            label=f"Affine Grid Sampler 2d",
            sub_label=f"bicubic f32, CL, BS={n}",
            description=f"Inductor (Torch {torch.__version__})",
        ).blocked_autorange(min_run_time=min_run_time)
    )
    compare = benchmark.Compare(results)
    compare.print()
