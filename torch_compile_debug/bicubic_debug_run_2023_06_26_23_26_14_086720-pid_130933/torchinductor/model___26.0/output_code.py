
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

import cv2

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()


cpp_fused_upsample_bicubic2d_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/mq/cmqzxwuyo7ryvun3egqos5jq5ak4fue7d2jbopbqs7pgpkhdpfh4.h"
extern "C" void kernel(const float* in_ptr0, float* out_ptr0)
{

    using index_t = long;

    {
        auto tmp3 = static_cast<float>(1.0);
        auto tmp5 = static_cast<float>(0.0);
        auto tmp22 = static_cast<index_t>(31);
        auto tmp24 = static_cast<index_t>(0);
        auto tmp39 = static_cast<float>(-0.75);
        auto tmp41 = static_cast<float>(-3.75);
        auto tmp44 = static_cast<float>(-6.0);
        auto tmp47 = static_cast<float>(-3.0);
        auto tmp49 = static_cast<float>(1.25);
        auto tmp51 = static_cast<float>(2.25);

        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(3L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(12L); i1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i2=static_cast<long>(0L); i2<static_cast<long>(12L); i2+=static_cast<long>(1L))
                {
                    // x = scale_w * (i2 + 0.5) - 0.5
                    auto tmp0 = static_cast<float>(0.833333333333333 + (2.66666666666667*i2));

                    auto tmp1 = std::floor(tmp0);
                    auto tmp2 = tmp0 - tmp1;
                    auto tmp4 = min_propagate_nan(tmp3, tmp2);
                    auto tmp6 = max_propagate_nan(tmp5, tmp4);

                    auto tmp7 = static_cast<float>(0.833333333333333 + (2.66666666666667*i1));

                    auto tmp8 = std::floor(tmp7);
                    auto tmp9 = tmp7 - tmp8;
                    auto tmp10 = min_propagate_nan(tmp3, tmp9);
                    auto tmp11 = max_propagate_nan(tmp5, tmp10);

                    auto tmp12 = static_cast<index_t>(tmp8);
                    auto tmp13 = static_cast<index_t>(tmp1);

                    auto tmp14 = tmp12 + -1;
                    auto tmp15 = tmp12 + 0;
                    auto tmp16 = tmp12 + 1;
                    auto tmp17 = tmp12 + 2;

                    auto tmp18 = tmp13 + -1;
                    auto tmp19 = tmp13 + 0;
                    auto tmp20 = tmp13 + 1;
                    auto tmp21 = tmp13 + 2;

                    auto tmp23 = min_propagate_nan(tmp22, tmp14);
                    auto tmp25 = max_propagate_nan(tmp24, tmp23);

                    auto tmp26 = min_propagate_nan(tmp22, tmp18);
                    auto tmp27 = max_propagate_nan(tmp24, tmp26);

                    auto tmp29 = min_propagate_nan(tmp22, tmp19);
                    auto tmp30 = max_propagate_nan(tmp24, tmp29);

                    auto tmp32 = min_propagate_nan(tmp22, tmp20);
                    auto tmp33 = max_propagate_nan(tmp24, tmp32);

                    auto tmp35 = min_propagate_nan(tmp22, tmp21);
                    auto tmp36 = max_propagate_nan(tmp24, tmp35);


                    auto tmp28 = in_ptr0[static_cast<long>(tmp27 + (32L*tmp25) + (1024L*i0))];
                    auto tmp31 = in_ptr0[static_cast<long>(tmp30 + (32L*tmp25) + (1024L*i0))];
                    auto tmp34 = in_ptr0[static_cast<long>(tmp33 + (32L*tmp25) + (1024L*i0))];
                    auto tmp37 = in_ptr0[static_cast<long>(tmp36 + (32L*tmp25) + (1024L*i0))];


                    auto tmp38 = tmp6 + tmp3;
                    auto tmp40 = decltype(tmp39)(tmp39 * tmp38);

                    auto tmp42 = tmp40 - tmp41;
                    auto tmp43 = decltype(tmp42)(tmp42 * tmp38);

                    auto tmp45 = tmp43 + tmp44;
                    auto tmp46 = decltype(tmp45)(tmp45 * tmp38);

                    auto tmp48 = tmp46 - tmp47;
                    auto tmp50 = decltype(tmp49)(tmp49 * tmp6);

                    auto tmp52 = tmp50 - tmp51;
                    auto tmp53 = decltype(tmp52)(tmp52 * tmp6);

                    auto tmp54 = decltype(tmp53)(tmp53 * tmp6);
                    auto tmp55 = tmp54 + tmp3;

                    auto tmp56 = tmp3 - tmp6;
                    auto tmp57 = decltype(tmp49)(tmp49 * tmp56);

                    auto tmp58 = tmp57 - tmp51;
                    auto tmp59 = decltype(tmp58)(tmp58 * tmp56);

                    auto tmp60 = decltype(tmp59)(tmp59 * tmp56);
                    auto tmp61 = tmp60 + tmp3;

                    auto tmp62 = tmp56 + tmp3;
                    auto tmp63 = decltype(tmp39)(tmp39 * tmp62);

                    auto tmp64 = tmp63 - tmp41;
                    auto tmp65 = decltype(tmp64)(tmp64 * tmp62);

                    auto tmp66 = tmp65 + tmp44;
                    auto tmp67 = decltype(tmp66)(tmp66 * tmp62);

                    auto tmp68 = tmp67 - tmp47;
                    auto tmp69 = decltype(tmp28)(tmp28 * tmp48);

                    auto tmp70 = decltype(tmp31)(tmp31 * tmp55);
                    auto tmp71 = tmp69 + tmp70;

                    auto tmp72 = decltype(tmp34)(tmp34 * tmp61);
                    auto tmp73 = tmp71 + tmp72;

                    auto tmp74 = decltype(tmp37)(tmp37 * tmp68);
                    auto tmp75 = tmp73 + tmp74;
                    auto tmp76 = min_propagate_nan(tmp22, tmp15);
                    auto tmp77 = max_propagate_nan(tmp24, tmp76);

                    auto tmp78 = in_ptr0[static_cast<long>(tmp27 + (32L*tmp77) + (1024L*i0))];
                    auto tmp79 = in_ptr0[static_cast<long>(tmp30 + (32L*tmp77) + (1024L*i0))];
                    auto tmp80 = in_ptr0[static_cast<long>(tmp33 + (32L*tmp77) + (1024L*i0))];
                    auto tmp81 = in_ptr0[static_cast<long>(tmp36 + (32L*tmp77) + (1024L*i0))];
                    auto tmp82 = decltype(tmp78)(tmp78 * tmp48);
                    auto tmp83 = decltype(tmp79)(tmp79 * tmp55);
                    auto tmp84 = tmp82 + tmp83;
                    auto tmp85 = decltype(tmp80)(tmp80 * tmp61);
                    auto tmp86 = tmp84 + tmp85;
                    auto tmp87 = decltype(tmp81)(tmp81 * tmp68);
                    auto tmp88 = tmp86 + tmp87;
                    auto tmp89 = min_propagate_nan(tmp22, tmp16);
                    auto tmp90 = max_propagate_nan(tmp24, tmp89);
                    auto tmp91 = in_ptr0[static_cast<long>(tmp27 + (32L*tmp90) + (1024L*i0))];
                    auto tmp92 = in_ptr0[static_cast<long>(tmp30 + (32L*tmp90) + (1024L*i0))];
                    auto tmp93 = in_ptr0[static_cast<long>(tmp33 + (32L*tmp90) + (1024L*i0))];
                    auto tmp94 = in_ptr0[static_cast<long>(tmp36 + (32L*tmp90) + (1024L*i0))];
                    auto tmp95 = decltype(tmp91)(tmp91 * tmp48);
                    auto tmp96 = decltype(tmp92)(tmp92 * tmp55);
                    auto tmp97 = tmp95 + tmp96;
                    auto tmp98 = decltype(tmp93)(tmp93 * tmp61);
                    auto tmp99 = tmp97 + tmp98;
                    auto tmp100 = decltype(tmp94)(tmp94 * tmp68);
                    auto tmp101 = tmp99 + tmp100;
                    auto tmp102 = min_propagate_nan(tmp22, tmp17);
                    auto tmp103 = max_propagate_nan(tmp24, tmp102);
                    auto tmp104 = in_ptr0[static_cast<long>(tmp27 + (32L*tmp103) + (1024L*i0))];
                    auto tmp105 = in_ptr0[static_cast<long>(tmp30 + (32L*tmp103) + (1024L*i0))];
                    auto tmp106 = in_ptr0[static_cast<long>(tmp33 + (32L*tmp103) + (1024L*i0))];
                    auto tmp107 = in_ptr0[static_cast<long>(tmp36 + (32L*tmp103) + (1024L*i0))];
                    auto tmp108 = decltype(tmp104)(tmp104 * tmp48);
                    auto tmp109 = decltype(tmp105)(tmp105 * tmp55);
                    auto tmp110 = tmp108 + tmp109;
                    auto tmp111 = decltype(tmp106)(tmp106 * tmp61);
                    auto tmp112 = tmp110 + tmp111;
                    auto tmp113 = decltype(tmp107)(tmp107 * tmp68);
                    auto tmp114 = tmp112 + tmp113;
                    auto tmp115 = tmp11 + tmp3;
                    auto tmp116 = decltype(tmp39)(tmp39 * tmp115);
                    auto tmp117 = tmp116 - tmp41;
                    auto tmp118 = decltype(tmp117)(tmp117 * tmp115);
                    auto tmp119 = tmp118 + tmp44;
                    auto tmp120 = decltype(tmp119)(tmp119 * tmp115);
                    auto tmp121 = tmp120 - tmp47;
                    auto tmp122 = decltype(tmp49)(tmp49 * tmp11);
                    auto tmp123 = tmp122 - tmp51;
                    auto tmp124 = decltype(tmp123)(tmp123 * tmp11);
                    auto tmp125 = decltype(tmp124)(tmp124 * tmp11);
                    auto tmp126 = tmp125 + tmp3;
                    auto tmp127 = tmp3 - tmp11;
                    auto tmp128 = decltype(tmp49)(tmp49 * tmp127);
                    auto tmp129 = tmp128 - tmp51;
                    auto tmp130 = decltype(tmp129)(tmp129 * tmp127);
                    auto tmp131 = decltype(tmp130)(tmp130 * tmp127);
                    auto tmp132 = tmp131 + tmp3;
                    auto tmp133 = tmp127 + tmp3;
                    auto tmp134 = decltype(tmp39)(tmp39 * tmp133);
                    auto tmp135 = tmp134 - tmp41;
                    auto tmp136 = decltype(tmp135)(tmp135 * tmp133);
                    auto tmp137 = tmp136 + tmp44;
                    auto tmp138 = decltype(tmp137)(tmp137 * tmp133);
                    auto tmp139 = tmp138 - tmp47;
                    auto tmp140 = decltype(tmp75)(tmp75 * tmp121);
                    auto tmp141 = decltype(tmp88)(tmp88 * tmp126);
                    auto tmp142 = tmp140 + tmp141;
                    auto tmp143 = decltype(tmp101)(tmp101 * tmp132);
                    auto tmp144 = tmp142 + tmp143;
                    auto tmp145 = decltype(tmp114)(tmp114 * tmp139);
                    auto tmp146 = tmp144 + tmp145;

                    out_ptr0[static_cast<long>(i2 + (12L*i1) + (144L*i0))] = tmp146;
                }
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (1, 3, 32, 32), (3072, 1024, 32, 1))

    buf0 = empty_strided((1, 3, 12, 12), (432, 144, 12, 1), device='cpu', dtype=torch.float32)

    cpp_fused_upsample_bicubic2d_0(c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()))
    del arg0_1
    return (buf0, )


# def benchmark_compiled_module(times=10, repeat=10):
#     from torch._dynamo.testing import rand_strided
#     from torch._inductor.utils import print_performance
#     arg0_1 = rand_strided((1, 3, 32, 32), (3072, 1024, 32, 1), device='cpu', dtype=torch.float32)
#     return print_performance(lambda: call([arg0_1]), times=times, repeat=repeat)


def func():
    device = "cpu"
    x = torch.arange(3 * 32 * 32, device=device).reshape(1, 3, 32, 32).to(torch.uint8).to(torch.float32)

    output = call([x, ])[0]
    expected = torch.nn.functional.interpolate(x, size=(12, 12), mode="bicubic", antialias=False)

    x_np = x[0].permute((1, 2, 0)).numpy()
    expected_cv2 = cv2.resize(x_np, (12, 12), interpolation=cv2.INTER_CUBIC)

    torch.set_printoptions(precision=6)
    print("output:\n", output[0, 0, :, :6])
    print("expected:\n", expected[0, 0, :, :6])
    print("opencv:\n", expected_cv2[:, :6, 0])

    # torch.set_printoptions(precision=5)
    m = (output - expected).abs() > 0
    # print(
    #     (output - expected)[0, 0, :5, :5]
    # )
    print(output[m][:10])
    print(expected[m][:10])

    torch.testing.assert_close(output, expected)
    # torch.testing.assert_close(output[:, :, 3:-3, 3:-3], expected[:, :, 3:-3, 3:-3])


if __name__ == "__main__":
    # from torch._inductor.utils import compiled_module_main
    # compiled_module_main('None', benchmark_compiled_module)
    func()
