
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


# kernel path: /tmp/torchinductor_root/ov/covemgjmswvgwikbzf2vw4nvlhx24yoobvter3cpyhonqwf2czy6.py
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


# Source Nodes: [grid_sample], Original ATen: [aten.grid_sampler_2d]
triton_poi_fused_grid_sampler_2d_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[1048576], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr1'], 'autotune_hints': set(), 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=())]})
@triton.jit
def triton_(in_out_ptr1, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    arg0_1, arg1_1, arg2_1 = args
    args.clear()
    s0 = arg0_1
    assert_size_stride(arg1_1, (s0, 3, 345, 456), (471960, 1, 1368, 3))
    assert_size_stride(arg2_1, (s0, 2, 3), (6, 3, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf1 = empty_strided((s0, 157320, 2), (314640, 2, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [affine_grid], Original ATen: [aten.affine_grid_generator]
        triton_poi_fused_affine_grid_generator_0_xnumel = 314640*s0
        stream0 = get_cuda_stream(0)
        triton_poi_fused_affine_grid_generator_0.run(arg2_1, buf1, triton_poi_fused_affine_grid_generator_0_xnumel, grid=grid(triton_poi_fused_affine_grid_generator_0_xnumel), stream=stream0)
        del arg2_1
        buf10 = empty_strided((s0, 3, 345, 456), (471960, 157320, 456, 1), device='cuda', dtype=torch.float32)
        buf11 = buf10; del buf10  # reuse
        buf16 = buf11; del buf11  # reuse
        buf62 = buf16; del buf16  # reuse
        # Source Nodes: [grid_sample], Original ATen: [aten.grid_sampler_2d]
        triton_poi_fused_grid_sampler_2d_1_xnumel = 471960*s0
        triton_poi_fused_grid_sampler_2d_1.run(buf62, buf1, arg1_1, triton_poi_fused_grid_sampler_2d_1_xnumel, grid=grid(triton_poi_fused_grid_sampler_2d_1_xnumel), stream=stream0)
        del arg1_1
        return (buf62, )


def benchmark_compiled_module(times=1000, repeat=20):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 2
    arg1_1 = rand_strided((2, 3, 345, 456), (471960, 1, 1368, 3), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((2, 2, 3), (6, 3, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
