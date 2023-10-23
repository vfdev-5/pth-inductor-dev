import torch
from torch import empty
from torch._inductor.triton_heuristics import grid
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers


# @persistent_reduction(
#     size_hints=[1048576, 32],
#     reduction_hint=ReductionHint.DEFAULT,
#     filename=__file__,
#     meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_index_mul_sum_0', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]}
# )

@triton.jit
def triton_per_fused_index_mul_sum_0(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x1 = (xindex // 256) % 256
    r4 = (rindex // 5)
    x0 = xindex % 256
    r3 = rindex % 5
    x2 = (xindex // 65536)
    x7 = xindex

    # One spatial dim
    tmp0 = x1    # (XBLOCK, 1)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.953125
    tmp5 = tmp3 * tmp4   # center
    tl.static_print("tmp5, center:", tmp5.shape)
    tmp6 = tmp5 + tmp4
    tmp7 = tmp6 + tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1, 1], 500, tl.int32)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tmp11 = tmp5 - tmp4
    tmp12 = tmp11 + tmp2
    tmp13 = tmp12.to(tl.int32)
    tmp14 = tl.full([1, 1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp13, tmp14)   # xmin
    tmp16 = tmp10 - tmp15                          # xsize
    tmp17 = tl.full([1, 1], 5, tl.int32)
    tmp18 = triton_helpers.minimum(tmp16, tmp17)   # xsize

    tl.static_print("1 center:", tmp5.shape)
    tl.static_print("1 xmin:", tmp15.shape)
    tl.static_print("1 xsize:", tmp18.shape)

    tmp19 = r4              # j
    tmp20 = tmp19 < tmp18   # j < xsize
    tmp21 = tmp19 + tmp15   # j + xmin
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp22 - tmp5    # j + xmin - center
    tmp24 = tmp23 + tmp2    # j + xmin - center + 0.5
    tmp25 = 0.512
    tmp26 = tmp24 * tmp25   # (j + xmin - center + 0.5) * invscale

    tmp27 = tl.abs(tmp26)
    tmp28 = 1.0
    tmp29 = triton_helpers.minimum(tmp27, tmp28)
    tmp30 = tmp28 - tmp29   # weights = aa_linear_filter(...)

    tmp31 = 0.0
    tmp32 = tl.where(tmp20, tmp30, tmp31)  # weights = torch.where(j < xsize, weights, 0.0)
    tl.static_print("1 non-normed weights:", tmp32.shape)


    # compute total_weights
    tmp33 = tmp14 < tmp18
    tmp34 = tmp14 + tmp15
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp35 - tmp5
    tmp37 = tmp36 + tmp2
    tmp38 = tmp37 * tmp25

    tmp39 = tl.abs(tmp38)
    tmp40 = triton_helpers.minimum(tmp39, tmp28)
    tmp41 = tmp28 - tmp40
    tmp42 = tl.where(tmp33, tmp41, tmp31)


    tmp43 = tl.full([1, 1], 1, tl.int32)
    tmp44 = tmp43 < tmp18
    tmp45 = tmp43 + tmp15
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp46 - tmp5
    tmp48 = tmp47 + tmp2
    tmp49 = tmp48 * tmp25
    tmp50 = tl.abs(tmp49)
    tmp51 = triton_helpers.minimum(tmp50, tmp28)
    tmp52 = tmp28 - tmp51
    tmp53 = tl.where(tmp44, tmp52, tmp31)

    tmp54 = tmp42 + tmp53

    tmp55 = tl.full([1, 1], 2, tl.int32)
    tmp56 = tmp55 < tmp18
    tmp57 = tmp55 + tmp15
    tmp58 = tmp57.to(tl.float32)
    tmp59 = tmp58 - tmp5
    tmp60 = tmp59 + tmp2
    tmp61 = tmp60 * tmp25
    tmp62 = tl.abs(tmp61)
    tmp63 = triton_helpers.minimum(tmp62, tmp28)
    tmp64 = tmp28 - tmp63
    tmp65 = tl.where(tmp56, tmp64, tmp31)

    tmp66 = tmp54 + tmp65

    tmp67 = tl.full([1, 1], 3, tl.int32)
    tmp68 = tmp67 < tmp18
    tmp69 = tmp67 + tmp15
    tmp70 = tmp69.to(tl.float32)
    tmp71 = tmp70 - tmp5
    tmp72 = tmp71 + tmp2
    tmp73 = tmp72 * tmp25
    tmp74 = tl.abs(tmp73)
    tmp75 = triton_helpers.minimum(tmp74, tmp28)
    tmp76 = tmp28 - tmp75
    tmp77 = tl.where(tmp68, tmp76, tmp31)

    tmp78 = tmp66 + tmp77

    tmp79 = tl.full([1, 1], 4, tl.int32)
    tmp80 = tmp79 < tmp18
    tmp81 = tmp79 + tmp15
    tmp82 = tmp81.to(tl.float32)
    tmp83 = tmp82 - tmp5
    tmp84 = tmp83 + tmp2
    tmp85 = tmp84 * tmp25
    tmp86 = tl.abs(tmp85)
    tmp87 = triton_helpers.minimum(tmp86, tmp28)
    tmp88 = tmp28 - tmp87
    tmp89 = tl.where(tmp80, tmp88, tmp31)

    tmp90 = tmp78 + tmp89

    tmp91 = tmp32 / tmp90  # weights = weights / total_weights
    tl.static_print("1 weights:", tmp91.shape)

    # Next spatial dimension
    tmp92 = x0
    tmp93 = tmp92.to(tl.float32)
    tmp94 = tmp93 + tmp2
    tmp95 = 1.5625
    tmp96 = tmp94 * tmp95
    tmp97 = tmp96 + tmp95
    tmp98 = tmp97 + tmp2
    tmp99 = tmp98.to(tl.int32)
    tmp100 = tl.full([1, 1], 400, tl.int32)
    tmp101 = triton_helpers.minimum(tmp99, tmp100)
    tmp102 = tmp96 - tmp95
    tmp103 = tmp102 + tmp2
    tmp104 = tmp103.to(tl.int32)
    tmp105 = triton_helpers.maximum(tmp104, tmp14)  # xmin
    tmp106 = tmp101 - tmp105
    tmp107 = triton_helpers.minimum(tmp106, tmp17)

    tl.static_print("2 center:", tmp96.shape)
    tl.static_print("2 xmin:", tmp105.shape)
    tl.static_print("2 xsize:", tmp107.shape)


    # compute weights
    tmp108 = r3
    tmp109 = tmp108 < tmp107
    tmp110 = tmp108 + tmp105
    tmp111 = tmp110.to(tl.float32)
    tmp112 = tmp111 - tmp96
    tmp113 = tmp112 + tmp2
    tmp114 = 0.64
    tmp115 = tmp113 * tmp114
    tmp116 = tl.abs(tmp115)
    tmp117 = triton_helpers.minimum(tmp116, tmp28)
    tmp118 = tmp28 - tmp117

    tmp119 = tl.where(tmp109, tmp118, tmp31)

    # compute total_weights
    tmp120 = tmp14 < tmp107
    tmp121 = tmp14 + tmp105
    tmp122 = tmp121.to(tl.float32)
    tmp123 = tmp122 - tmp96
    tmp124 = tmp123 + tmp2
    tmp125 = tmp124 * tmp114
    tmp126 = tl.abs(tmp125)
    tmp127 = triton_helpers.minimum(tmp126, tmp28)
    tmp128 = tmp28 - tmp127
    tmp129 = tl.where(tmp120, tmp128, tmp31)

    tmp130 = tmp43 < tmp107
    tmp131 = tmp43 + tmp105
    tmp132 = tmp131.to(tl.float32)
    tmp133 = tmp132 - tmp96
    tmp134 = tmp133 + tmp2
    tmp135 = tmp134 * tmp114
    tmp136 = tl.abs(tmp135)
    tmp137 = triton_helpers.minimum(tmp136, tmp28)
    tmp138 = tmp28 - tmp137
    tmp139 = tl.where(tmp130, tmp138, tmp31)

    tmp140 = tmp129 + tmp139

    tmp141 = tmp55 < tmp107
    tmp142 = tmp55 + tmp105
    tmp143 = tmp142.to(tl.float32)
    tmp144 = tmp143 - tmp96
    tmp145 = tmp144 + tmp2
    tmp146 = tmp145 * tmp114
    tmp147 = tl.abs(tmp146)
    tmp148 = triton_helpers.minimum(tmp147, tmp28)
    tmp149 = tmp28 - tmp148
    tmp150 = tl.where(tmp141, tmp149, tmp31)

    tmp151 = tmp140 + tmp150

    tmp152 = tmp67 < tmp107
    tmp153 = tmp67 + tmp105
    tmp154 = tmp153.to(tl.float32)
    tmp155 = tmp154 - tmp96
    tmp156 = tmp155 + tmp2
    tmp157 = tmp156 * tmp114
    tmp158 = tl.abs(tmp157)
    tmp159 = triton_helpers.minimum(tmp158, tmp28)
    tmp160 = tmp28 - tmp159
    tmp161 = tl.where(tmp152, tmp160, tmp31)

    tmp162 = tmp151 + tmp161

    tmp163 = tmp79 < tmp107
    tmp164 = tmp79 + tmp105
    tmp165 = tmp164.to(tl.float32)
    tmp166 = tmp165 - tmp96
    tmp167 = tmp166 + tmp2
    tmp168 = tmp167 * tmp114
    tmp169 = tl.abs(tmp168)
    tmp170 = triton_helpers.minimum(tmp169, tmp28)
    tmp171 = tmp28 - tmp170
    tmp172 = tl.where(tmp163, tmp171, tmp31)

    tmp173 = tmp162 + tmp172


    tmp174 = tmp119 / tmp173
    tl.static_print("2 weights:", tmp174.shape)


    tmp175 = tmp15.to(tl.int32)   # xmin  Y-axis
    tmp176 = tmp175 + tmp19
    tmp177 = tl.full([1, 1], 499, tl.int64)
    tmp178 = triton_helpers.minimum(tmp176, tmp177)  # y_indices: (XBLOCK, 32)
    tl.static_print("y_indices:", tmp178.shape)

    tmp179 = tmp105.to(tl.int32)  # xmin  X-axis
    tmp180 = tmp179 + tmp108
    tmp181 = tl.full([1, 1], 399, tl.int64)
    tmp182 = triton_helpers.minimum(tmp180, tmp181)  # x_indices: (XBLOCK, 32)
    tl.static_print("x_indices:", tmp182.shape)

    # input_selected = input[:, :, y_indices, x_indices]
    # tmp183 = tl.load(in_ptr0 + (tmp182 + (400*tmp178) + (200000*x2)), rmask, eviction_policy='evict_last', other=0)
    tmp183 = tl.load(in_ptr0 + (tmp182 + (400*tmp178) + (200000*x2)), rmask, other=0)
    tl.static_print("input_selected:", tmp183.shape)

    # tmp183: N, R

    # tmp183[0, :]
    # S[i,j] S[i,j + 1], S[i,j + 2] ... S[i,j + 4]
    # S[i + 1,j] S[i + 1,j + 1], S[i,j + 2] ... S[i,j + 4]
    # S[i + 2,j] S[i,j + 1], S[i,j + 2] ... S[i,j + 4]
    # S[i,j] S[i,j + 1], S[i,j + 2] ... S[i,j + 4]

    # tmp174:
    #  hw0 hw1 hw2 ... hw4
    #  hw0 hw1 hw2 ... hw4
    #  hw0 hw1 hw2 ... hw4
    #  hw0 hw1 hw2 ... hw4
    #  hw0 hw1 hw2 ... hw4

    # tmp91:
    #  vw0 vw0 vw0 vw0 vw0
    #  vw1 vw1 vw1 vw1 vw1
    #  ...
    #  vw4 vw4 vw4 vw4 vw4

    # tmp174: 1, R
    # tmp91: 1, R

    tmp184 = tmp174 * tmp183
    tmp185 = tmp91 * tmp184

    tl.static_print("before broadcast_to:", tmp185.shape)
    # tmp186 = tl.broadcast_to(tmp185, [XBLOCK, RBLOCK])
    tmp186 = tmp185
    tl.static_print("after broadcast_to:", tmp186.shape)

    # tmp188 = tl.where(rmask, tmp186, 0)
    tmp188 = tmp186
    tmp189 = tl.sum(tmp188, 1)[:, None]
    tl.store(out_ptr0 + (x7), tmp189, None)


@triton.jit
def triton_per_fused_index_mul_sum_0_v2(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 786432 // 12
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel

    x1 = (xindex // 256) % 256
    r4 = (rindex // 5)
    x0 = xindex % 256
    r3 = rindex % 5

    x2 = (xindex // 65536)
    x7 = xindex

    # One spatial dim
    tmp0 = x1    # (XBLOCK, 1)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.953125
    tmp5 = tmp3 * tmp4   # center
    tl.static_print("tmp5, center:", tmp5.shape)
    tmp6 = tmp5 + tmp4
    tmp7 = tmp6 + tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1, 1], 500, tl.int32)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tmp11 = tmp5 - tmp4
    tmp12 = tmp11 + tmp2
    tmp13 = tmp12.to(tl.int32)
    tmp14 = tl.full([1, 1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp13, tmp14)   # xmin
    tmp16 = tmp10 - tmp15                          # xsize
    tmp17 = tl.full([1, 1], 5, tl.int32)
    tmp18 = triton_helpers.minimum(tmp16, tmp17)   # xsize

    tl.static_print("1 center:", tmp5.shape)
    tl.static_print("1 xmin:", tmp15.shape)
    tl.static_print("1 xsize:", tmp18.shape)

    tmp19 = r4              # j
    tmp20 = tmp19 < tmp18   # j < xsize
    tmp21 = tmp19 + tmp15   # j + xmin
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp22 - tmp5    # j + xmin - center
    tmp24 = tmp23 + tmp2    # j + xmin - center + 0.5
    tmp25 = 0.512
    tmp26 = tmp24 * tmp25   # (j + xmin - center + 0.5) * invscale

    tmp27 = tl.abs(tmp26)
    tmp28 = 1.0
    tmp29 = triton_helpers.minimum(tmp27, tmp28)
    tmp30 = tmp28 - tmp29   # weights = aa_linear_filter(...)

    tmp31 = 0.0
    tmp32 = tl.where(tmp20, tmp30, tmp31)  # weights = torch.where(j < xsize, weights, 0.0)
    tl.static_print("1 non-normed weights:", tmp32.shape)


    # compute total_weights
    tmp33 = tmp14 < tmp18
    tmp34 = tmp14 + tmp15
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp35 - tmp5
    tmp37 = tmp36 + tmp2
    tmp38 = tmp37 * tmp25

    tmp39 = tl.abs(tmp38)
    tmp40 = triton_helpers.minimum(tmp39, tmp28)
    tmp41 = tmp28 - tmp40
    tmp42 = tl.where(tmp33, tmp41, tmp31)


    tmp43 = tl.full([1, 1], 1, tl.int32)
    tmp44 = tmp43 < tmp18
    tmp45 = tmp43 + tmp15
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp46 - tmp5
    tmp48 = tmp47 + tmp2
    tmp49 = tmp48 * tmp25
    tmp50 = tl.abs(tmp49)
    tmp51 = triton_helpers.minimum(tmp50, tmp28)
    tmp52 = tmp28 - tmp51
    tmp53 = tl.where(tmp44, tmp52, tmp31)

    tmp54 = tmp42 + tmp53

    tmp55 = tl.full([1, 1], 2, tl.int32)
    tmp56 = tmp55 < tmp18
    tmp57 = tmp55 + tmp15
    tmp58 = tmp57.to(tl.float32)
    tmp59 = tmp58 - tmp5
    tmp60 = tmp59 + tmp2
    tmp61 = tmp60 * tmp25
    tmp62 = tl.abs(tmp61)
    tmp63 = triton_helpers.minimum(tmp62, tmp28)
    tmp64 = tmp28 - tmp63
    tmp65 = tl.where(tmp56, tmp64, tmp31)

    tmp66 = tmp54 + tmp65

    tmp67 = tl.full([1, 1], 3, tl.int32)
    tmp68 = tmp67 < tmp18
    tmp69 = tmp67 + tmp15
    tmp70 = tmp69.to(tl.float32)
    tmp71 = tmp70 - tmp5
    tmp72 = tmp71 + tmp2
    tmp73 = tmp72 * tmp25
    tmp74 = tl.abs(tmp73)
    tmp75 = triton_helpers.minimum(tmp74, tmp28)
    tmp76 = tmp28 - tmp75
    tmp77 = tl.where(tmp68, tmp76, tmp31)

    tmp78 = tmp66 + tmp77

    tmp79 = tl.full([1, 1], 4, tl.int32)
    tmp80 = tmp79 < tmp18
    tmp81 = tmp79 + tmp15
    tmp82 = tmp81.to(tl.float32)
    tmp83 = tmp82 - tmp5
    tmp84 = tmp83 + tmp2
    tmp85 = tmp84 * tmp25
    tmp86 = tl.abs(tmp85)
    tmp87 = triton_helpers.minimum(tmp86, tmp28)
    tmp88 = tmp28 - tmp87
    tmp89 = tl.where(tmp80, tmp88, tmp31)

    tmp90 = tmp78 + tmp89

    tmp91 = tmp32 / tmp90  # weights = weights / total_weights
    tl.static_print("1 weights:", tmp91.shape)

    # Next spatial dimension
    tmp92 = x0
    tmp93 = tmp92.to(tl.float32)
    tmp94 = tmp93 + tmp2
    tmp95 = 1.5625
    tmp96 = tmp94 * tmp95
    tmp97 = tmp96 + tmp95
    tmp98 = tmp97 + tmp2
    tmp99 = tmp98.to(tl.int32)
    tmp100 = tl.full([1, 1], 400, tl.int32)
    tmp101 = triton_helpers.minimum(tmp99, tmp100)
    tmp102 = tmp96 - tmp95
    tmp103 = tmp102 + tmp2
    tmp104 = tmp103.to(tl.int32)
    tmp105 = triton_helpers.maximum(tmp104, tmp14)  # xmin
    tmp106 = tmp101 - tmp105
    tmp107 = triton_helpers.minimum(tmp106, tmp17)

    tl.static_print("2 center:", tmp96.shape)
    tl.static_print("2 xmin:", tmp105.shape)
    tl.static_print("2 xsize:", tmp107.shape)


    # compute weights
    tmp108 = r3
    tmp109 = tmp108 < tmp107
    tmp110 = tmp108 + tmp105
    tmp111 = tmp110.to(tl.float32)
    tmp112 = tmp111 - tmp96
    tmp113 = tmp112 + tmp2
    tmp114 = 0.64
    tmp115 = tmp113 * tmp114
    tmp116 = tl.abs(tmp115)
    tmp117 = triton_helpers.minimum(tmp116, tmp28)
    tmp118 = tmp28 - tmp117

    tmp119 = tl.where(tmp109, tmp118, tmp31)

    # compute total_weights
    tmp120 = tmp14 < tmp107
    tmp121 = tmp14 + tmp105
    tmp122 = tmp121.to(tl.float32)
    tmp123 = tmp122 - tmp96
    tmp124 = tmp123 + tmp2
    tmp125 = tmp124 * tmp114
    tmp126 = tl.abs(tmp125)
    tmp127 = triton_helpers.minimum(tmp126, tmp28)
    tmp128 = tmp28 - tmp127
    tmp129 = tl.where(tmp120, tmp128, tmp31)

    tmp130 = tmp43 < tmp107
    tmp131 = tmp43 + tmp105
    tmp132 = tmp131.to(tl.float32)
    tmp133 = tmp132 - tmp96
    tmp134 = tmp133 + tmp2
    tmp135 = tmp134 * tmp114
    tmp136 = tl.abs(tmp135)
    tmp137 = triton_helpers.minimum(tmp136, tmp28)
    tmp138 = tmp28 - tmp137
    tmp139 = tl.where(tmp130, tmp138, tmp31)

    tmp140 = tmp129 + tmp139

    tmp141 = tmp55 < tmp107
    tmp142 = tmp55 + tmp105
    tmp143 = tmp142.to(tl.float32)
    tmp144 = tmp143 - tmp96
    tmp145 = tmp144 + tmp2
    tmp146 = tmp145 * tmp114
    tmp147 = tl.abs(tmp146)
    tmp148 = triton_helpers.minimum(tmp147, tmp28)
    tmp149 = tmp28 - tmp148
    tmp150 = tl.where(tmp141, tmp149, tmp31)

    tmp151 = tmp140 + tmp150

    tmp152 = tmp67 < tmp107
    tmp153 = tmp67 + tmp105
    tmp154 = tmp153.to(tl.float32)
    tmp155 = tmp154 - tmp96
    tmp156 = tmp155 + tmp2
    tmp157 = tmp156 * tmp114
    tmp158 = tl.abs(tmp157)
    tmp159 = triton_helpers.minimum(tmp158, tmp28)
    tmp160 = tmp28 - tmp159
    tmp161 = tl.where(tmp152, tmp160, tmp31)

    tmp162 = tmp151 + tmp161

    tmp163 = tmp79 < tmp107
    tmp164 = tmp79 + tmp105
    tmp165 = tmp164.to(tl.float32)
    tmp166 = tmp165 - tmp96
    tmp167 = tmp166 + tmp2
    tmp168 = tmp167 * tmp114
    tmp169 = tl.abs(tmp168)
    tmp170 = triton_helpers.minimum(tmp169, tmp28)
    tmp171 = tmp28 - tmp170
    tmp172 = tl.where(tmp163, tmp171, tmp31)

    tmp173 = tmp162 + tmp172


    tmp174 = tmp119 / tmp173
    tl.static_print("2 weights:", tmp174.shape)


    tmp175 = tmp15.to(tl.int32)   # xmin  Y-axis
    tmp176 = tmp175 + tmp19
    tmp177 = tl.full([1, 1], 499, tl.int64)
    tmp178 = triton_helpers.minimum(tmp176, tmp177)  # y_indices: (XBLOCK, 32)
    tl.static_print("y_indices:", tmp178.shape)

    tmp179 = tmp105.to(tl.int32)  # xmin  X-axis
    tmp180 = tmp179 + tmp108
    tmp181 = tl.full([1, 1], 399, tl.int64)
    tmp182 = triton_helpers.minimum(tmp180, tmp181)  # x_indices: (XBLOCK, 32)
    tl.static_print("x_indices:", tmp182.shape)

    # input_selected = input[:, :, y_indices, x_indices]

    for i in range(12):
        # x2 = (xindex // 65536)
        # 65536 = 256 * 256 <= output
        # 200000 = 500 * 400 <= input

        x22 = x2 + i

        tmp183 = tl.load(in_ptr0 + (tmp182 + (400*tmp178) + (200000*x22)), rmask, eviction_policy='evict_last', other=0)
        tl.static_print("input_selected:", tmp183.shape)

        tmp184 = tmp174 * tmp183
        tmp185 = tmp91 * tmp184

        tl.static_print("before broadcast_to:", tmp185.shape)
        # tmp186 = tl.broadcast_to(tmp185, [XBLOCK, RBLOCK])
        tmp186 = tmp185
        tl.static_print("after broadcast_to:", tmp186.shape)

        # tmp188 = tl.where(rmask, tmp186, 0)
        tmp188 = tmp186
        tmp189 = tl.sum(tmp188, 1)[:, None]

        # x7 = xindex
        x77 = x7 + 65536 * i
        tl.store(out_ptr0 + x77, tmp189, None)


# def call(args):
#     arg0_1, = args
#     args.clear()
#     # assert_size_stride(arg0_1, (4, 3, 500, 400), (600000, 200000, 400, 1))
#     with torch.cuda._DeviceGuard(0):
#         torch.cuda.set_device(0) # no-op to ensure context
#         buf6 = empty((4, 3, 256, 256), device='cuda', dtype=torch.float32)
#         # Source Nodes: [img], Original ATen: [aten.index, aten.mul, aten.sum]
#         stream0 = get_cuda_stream(0)
#         triton_per_fused_index_mul_sum_0.run(arg0_1, buf6, 786432, 25, grid=grid(786432), stream=stream0)
#         del arg0_1
#         return (buf6, )


if __name__ == "__main__":

    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((4, 3, 500, 400), (600000, 200000, 400, 1), device='cuda:0', dtype=torch.float32)

    # out = call([arg0_1])
    # print(out[0].shape)

    fn = triton_per_fused_index_mul_sum_0
    grid_size = 786432

    # fn = triton_per_fused_index_mul_sum_0_v2
    # grid_size = 786432 // 12

    # with torch.cuda._DeviceGuard(0):
    #     torch.cuda.set_device(0) # no-op to ensure context
    #     output = empty((4, 3, 256, 256), device='cuda', dtype=torch.float32)

    #     grid = lambda meta: (triton.cdiv(grid_size, meta['XBLOCK']),)
    #     # NOTE:
    #     #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #     #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #     #  - Don't forget to pass meta-parameters as keywords arguments.
    #     fn[grid](arg0_1, output, 0, 0, XBLOCK=128)

    # expected = torch.nn.functional.interpolate(arg0_1, (256, 256), mode="bilinear", antialias=True, align_corners=False)
    # torch.testing.assert_close(expected, output)

    # Inspect Triton IR
    ret = triton.compile(fn, signature="*fp32,*fp32,i32,i32", constants={"XBLOCK": 256})

    # print("\n\n----------- ttir")
    # print(ret.asm["ttir"])
    # print("\n\n----------- ttgir")
    # print(ret.asm["ttgir"])
    # print("\n\n----------- ptx")
    # print(ret.asm["ptx"])
    # print("\n\n-----------")
    # print(ret.asm.keys())



## Example: https://github.com/openai/triton/blob/main/python/examples/copy_strided.py

# import triton
# import triton.language as tl


# # triton kernel
# @triton.jit
# def kernel(X, stride_xm,
#            Z, stride_zn,
#            BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
#     off_m = tl.arange(0, BLOCK_M)
#     off_n = tl.arange(0, BLOCK_N)
#     Xs = X + off_m[:, None] * stride_xm + off_n[None, :] * 1
#     Zs = Z + off_m[:, None] * 1 + off_n[None, :] * stride_zn
#     tl.store(Zs, tl.load(Xs))


# ret = triton.compile(kernel, signature="*fp32,i32,*fp32,i32", constants={"BLOCK_M": 64, "BLOCK_N": 64})
# print(ret.asm["ttgir"])