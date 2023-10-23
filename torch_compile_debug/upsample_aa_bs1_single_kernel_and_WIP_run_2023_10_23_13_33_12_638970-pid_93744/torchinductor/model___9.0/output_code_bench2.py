
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


# kernel path: /tmp/torchinductor_root/ue/cue3sarbzzoktyx4mnl5slwo7nh4vxx4a4vmu5fn5mr6dynmcj3c.py
# Source Nodes: [img], Original ATen: [aten.index, aten.mul, aten.sum]
# img => index, mul_4, mul_5, sum_3
triton_per_fused_index_mul_sum_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

print("Compile triton_per_fused_index_mul_sum_0")

@persistent_reduction(
    size_hints=[262144, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_index_mul_sum_0', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
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
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 + tmp2
    tmp4 = 1.953125
    tmp5 = tmp3 * tmp4
    tmp6 = tmp5 + tmp4
    tmp7 = tmp6 + tmp2
    tmp8 = tmp7.to(tl.int32)
    tmp9 = tl.full([1, 1], 500, tl.int32)
    tmp10 = triton_helpers.minimum(tmp8, tmp9)
    tmp11 = tmp5 - tmp4
    tmp12 = tmp11 + tmp2
    tmp13 = tmp12.to(tl.int32)
    tmp14 = tl.full([1, 1], 0, tl.int32)
    tmp15 = triton_helpers.maximum(tmp13, tmp14)
    tmp16 = tmp10 - tmp15
    tmp17 = tl.full([1, 1], 5, tl.int32)
    tmp18 = triton_helpers.minimum(tmp16, tmp17)
    tmp19 = r4
    tmp20 = tmp19 < tmp18
    tmp21 = tmp19 + tmp15
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp22 - tmp5
    tmp24 = tmp23 + tmp2
    tmp25 = 0.512
    tmp26 = tmp24 * tmp25
    tmp27 = tl.abs(tmp26)
    tmp28 = 1.0
    tmp29 = triton_helpers.minimum(tmp27, tmp28)
    tmp30 = tmp28 - tmp29
    tmp31 = 0.0
    tmp32 = tl.where(tmp20, tmp30, tmp31)
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
    tmp91 = tmp32 / tmp90
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
    tmp105 = triton_helpers.maximum(tmp104, tmp14)
    tmp106 = tmp101 - tmp105
    tmp107 = triton_helpers.minimum(tmp106, tmp17)
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
    tmp175 = tmp15.to(tl.int32)
    tmp176 = tmp175 + tmp19
    tmp177 = tl.full([1, 1], 499, tl.int64)
    tmp178 = triton_helpers.minimum(tmp176, tmp177)
    tmp179 = tmp105.to(tl.int32)
    tmp180 = tmp179 + tmp108
    tmp181 = tl.full([1, 1], 399, tl.int64)
    tmp182 = triton_helpers.minimum(tmp180, tmp181)

    tmp183 = tl.load(in_ptr0 + (tmp182 + (400*tmp178) + (200000*x2)), rmask, eviction_policy='evict_last', other=0)

    tmp184 = tmp174 * tmp183
    tmp185 = tmp91 * tmp184
    tmp186 = tl.broadcast_to(tmp185, [XBLOCK, RBLOCK])
    tmp188 = tl.where(rmask, tmp186, 0)
    tmp189 = tl.sum(tmp188, 1)[:, None]
    tl.store(out_ptr0 + (x7), tmp189, None)
''')


# kernel path: /tmp/torchinductor_root/pn/cpnj2zedw5fw5jfx5jz3jus7mapovlmf7xns5l27z7zchaknsfxp.py
# Source Nodes: [img], Original ATen: [aten.index, aten.mul, aten.sum]
# img => index, mul_4, mul_5, sum_3
triton_per_fused_index_mul_sum_0_v2 = async_compile.triton('triton_3', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

print("Compile triton_per_fused_index_mul_sum_0_v22")

@persistent_reduction(
    size_hints=[65536, 32],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_index_mul_sum_0', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]}
)
@triton.jit
def triton_3(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 196608
    rnumel = 25
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK

    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel

    # rindex = tl.arange(0, RBLOCK)[None, :]
    # rmask = rindex < rnumel
    # x1 = (xindex // 256) % 256
    # r4 = (rindex // 5)
    # x0 = xindex % 256
    # r3 = rindex % 5
    # x2 = (xindex // 65536)

    x7 = xindex

    # tmp0 = x1
    # tmp1 = tmp0.to(tl.float32)
    # tmp2 = 0.5
    # tmp3 = tmp1 + tmp2
    # tmp4 = 1.953125
    # tmp5 = tmp3 * tmp4
    # tmp6 = tmp5 + tmp4
    # tmp7 = tmp6 + tmp2
    # tmp8 = tmp7.to(tl.int32)
    # tmp9 = tl.full([1, 1], 500, tl.int32)
    # tmp10 = triton_helpers.minimum(tmp8, tmp9)
    # tmp11 = tmp5 - tmp4
    # tmp12 = tmp11 + tmp2
    # tmp13 = tmp12.to(tl.int32)
    # tmp14 = tl.full([1, 1], 0, tl.int32)
    # tmp15 = triton_helpers.maximum(tmp13, tmp14)
    # tmp16 = tmp10 - tmp15
    # tmp17 = tl.full([1, 1], 5, tl.int32)
    # tmp18 = triton_helpers.minimum(tmp16, tmp17)


    # tmp19 = r4
    # tmp32 = tl.full([1, RBLOCK], 1.0, tl.float32)



    # tmp92 = x0
    # tmp93 = tmp92.to(tl.float32)
    # tmp94 = tmp93 + tmp2
    # tmp95 = 1.5625
    # tmp96 = tmp94 * tmp95
    # tmp97 = tmp96 + tmp95
    # tmp98 = tmp97 + tmp2
    # tmp99 = tmp98.to(tl.int32)
    # tmp100 = tl.full([1, 1], 400, tl.int32)
    # tmp101 = triton_helpers.minimum(tmp99, tmp100)
    # tmp102 = tmp96 - tmp95
    # tmp103 = tmp102 + tmp2
    # tmp104 = tmp103.to(tl.int32)
    # tmp105 = triton_helpers.maximum(tmp104, tmp14)
    # tmp106 = tmp101 - tmp105
    # tmp107 = triton_helpers.minimum(tmp106, tmp17)


    # tmp108 = r3
    # tmp119 = tl.full([1, RBLOCK], 1.0, tl.float32)


    # tmp175 = tmp15.to(tl.int32)
    # tmp176 = tmp175 + tmp19
    # tmp177 = tl.full([1, 1], 499, tl.int64)
    # tmp178 = triton_helpers.minimum(tmp176, tmp177)
    # tmp179 = tmp105.to(tl.int32)
    # tmp180 = tmp179 + tmp108
    # tmp181 = tl.full([1, 1], 399, tl.int64)
    # tmp182 = triton_helpers.minimum(tmp180, tmp181)

    # # tmp183 = tl.load(in_ptr0 + (tmp182 + (400*tmp178) + (200000*x2)), rmask, eviction_policy='evict_last', other=0)
    # tmp183 = tl.full([XBLOCK, RBLOCK], 1.0, tl.float32)

    # # tmp184 = tmp119 * tmp183
    # # tmp185 = tmp32 * tmp184
    # tmp185 = tmp183

    # tmp186 = tl.broadcast_to(tmp185, [XBLOCK, RBLOCK])
    # tmp188 = tl.where(rmask, tmp186, 0)
    # tmp189 = tl.sum(tmp188, 1)[:, None]

    tmp189 = tl.full([XBLOCK, 1], 1.0, tl.float32)
    tl.store(out_ptr0 + (x7), tmp189, None)
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
        buf6 = empty((1, 3, 256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [img], Original ATen: [aten.index, aten.mul, aten.sum]
        stream0 = get_cuda_stream(0)
        triton_per_fused_index_mul_sum_0.run(arg0_1, buf6, 196608, 25, grid=grid(196608), stream=stream0)
        del arg0_1
        return (buf6, )


def call_v2(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 3, 500, 400), (600000, 200000, 400, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf6 = empty((1, 3, 256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [img], Original ATen: [aten.index, aten.mul, aten.sum]
        stream0 = get_cuda_stream(0)

        triton_per_fused_index_mul_sum_0_v2.run(arg0_1, buf6, 256 * 256 * 3, 25, grid=grid(256 * 256 * 3), stream=stream0)

        del arg0_1
        return (buf6, )


def ref_call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 3, 500, 400), (600000, 200000, 400, 1))
    output = torch.nn.functional.interpolate(arg0_1, (256, 256), mode="bilinear", antialias=True, align_corners=False)
    del arg0_1
    return (output, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((1, 3, 500, 400), (600000, 200000, 400, 1), device='cuda:0', dtype=torch.float32)
    expected = torch.nn.functional.interpolate(arg0_1, (256, 256), mode="bilinear", antialias=True, align_corners=False)

    output = call([arg0_1])[0]
    # print("- Check consistency v0")
    # torch.testing.assert_close(expected, output)
    print(output.min(), output.max(), output.mean())

    output = call_v2([arg0_1])[0]
    # print("- Check consistency v2")
    # # torch.testing.assert_close(expected, output)
    print(output.shape)
    print(output[output < 1].shape)
    print(output[0, 0, :5, :5])
    assert output.max().item() == output.min().item() == 1, (output.min(), output.max(), output.mean())


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
            label=f"Interpolate bilinear, AA=true, cuda",
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
            label=f"Interpolate bilinear, AA=true, cuda",
            sub_label=f"Input (1, 3, 500, 400) -> 256, 256, {arg0_1.dtype}, CF",
            description=f"Compiled",
        ).blocked_autorange(min_run_time=min_run_time)
    )
    results.append(
        benchmark.Timer(
            stmt=f"fn([x])",
            globals={
                "fn": call_v2,
                "x": arg0_1,
            },
            num_threads=torch.get_num_threads(),
            label=f"Interpolate bilinear, AA=true, cuda",
            sub_label=f"Input (1, 3, 500, 400) -> 256, 256, {arg0_1.dtype}, CF",
            description=f"Compiled v2",
        ).blocked_autorange(min_run_time=min_run_time)
    )

    compare = benchmark.Compare(results)
    compare.print()
