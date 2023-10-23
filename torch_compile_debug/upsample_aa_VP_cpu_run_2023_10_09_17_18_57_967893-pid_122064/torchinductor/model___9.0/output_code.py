
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


cpp_fused_clone_index_mul_sum_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/bf/cbf262yqjxhzxmw7lov36xiiezas3czyjs7cdvyrvlrje4xcl2kd.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(271L); x2+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(456L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = c10::convert<float>(tmp0);
                        auto tmp2 = c10::convert<float>(0.5);
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = c10::convert<float>(1.2730627306273063);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = tmp5 + tmp4;
                        auto tmp7 = tmp6 + tmp2;
                        auto tmp8 = c10::convert<long>(tmp7);
                        auto tmp9 = c10::convert<long>(345);
                        auto tmp10 = min_propagate_nan(tmp8, tmp9);
                        auto tmp11 = tmp5 - tmp4;
                        auto tmp12 = tmp11 + tmp2;
                        auto tmp13 = c10::convert<long>(tmp12);
                        auto tmp14 = c10::convert<long>(0);
                        auto tmp15 = max_propagate_nan(tmp13, tmp14);
                        auto tmp16 = tmp10 - tmp15;
                        auto tmp17 = c10::convert<long>(5);
                        auto tmp18 = min_propagate_nan(tmp16, tmp17);
                        auto tmp19 = tmp14 < tmp18;
                        auto tmp20 = tmp14 + tmp15;
                        auto tmp21 = c10::convert<float>(tmp20);
                        auto tmp22 = tmp21 - tmp5;
                        auto tmp23 = tmp22 + tmp2;
                        auto tmp24 = c10::convert<float>(0.7855072463768116);
                        auto tmp25 = decltype(tmp23)(tmp23 * tmp24);
                        auto tmp26 = std::abs(tmp25);
                        auto tmp27 = c10::convert<float>(1.0);
                        auto tmp28 = min_propagate_nan(tmp26, tmp27);
                        auto tmp29 = tmp27 - tmp28;
                        auto tmp30 = c10::convert<float>(0.0);
                        auto tmp31 = tmp19 ? tmp29 : tmp30;
                        auto tmp32 = c10::convert<long>(1);
                        auto tmp33 = tmp32 < tmp18;
                        auto tmp34 = tmp32 + tmp15;
                        auto tmp35 = c10::convert<float>(tmp34);
                        auto tmp36 = tmp35 - tmp5;
                        auto tmp37 = tmp36 + tmp2;
                        auto tmp38 = decltype(tmp37)(tmp37 * tmp24);
                        auto tmp39 = std::abs(tmp38);
                        auto tmp40 = min_propagate_nan(tmp39, tmp27);
                        auto tmp41 = tmp27 - tmp40;
                        auto tmp42 = tmp33 ? tmp41 : tmp30;
                        auto tmp43 = tmp31 + tmp42;
                        auto tmp44 = c10::convert<long>(2);
                        auto tmp45 = tmp44 < tmp18;
                        auto tmp46 = tmp44 + tmp15;
                        auto tmp47 = c10::convert<float>(tmp46);
                        auto tmp48 = tmp47 - tmp5;
                        auto tmp49 = tmp48 + tmp2;
                        auto tmp50 = decltype(tmp49)(tmp49 * tmp24);
                        auto tmp51 = std::abs(tmp50);
                        auto tmp52 = min_propagate_nan(tmp51, tmp27);
                        auto tmp53 = tmp27 - tmp52;
                        auto tmp54 = tmp45 ? tmp53 : tmp30;
                        auto tmp55 = tmp43 + tmp54;
                        auto tmp56 = c10::convert<long>(3);
                        auto tmp57 = tmp56 < tmp18;
                        auto tmp58 = tmp56 + tmp15;
                        auto tmp59 = c10::convert<float>(tmp58);
                        auto tmp60 = tmp59 - tmp5;
                        auto tmp61 = tmp60 + tmp2;
                        auto tmp62 = decltype(tmp61)(tmp61 * tmp24);
                        auto tmp63 = std::abs(tmp62);
                        auto tmp64 = min_propagate_nan(tmp63, tmp27);
                        auto tmp65 = tmp27 - tmp64;
                        auto tmp66 = tmp57 ? tmp65 : tmp30;
                        auto tmp67 = tmp55 + tmp66;
                        auto tmp68 = c10::convert<long>(4);
                        auto tmp69 = tmp68 < tmp18;
                        auto tmp70 = tmp68 + tmp15;
                        auto tmp71 = c10::convert<float>(tmp70);
                        auto tmp72 = tmp71 - tmp5;
                        auto tmp73 = tmp72 + tmp2;
                        auto tmp74 = decltype(tmp73)(tmp73 * tmp24);
                        auto tmp75 = std::abs(tmp74);
                        auto tmp76 = min_propagate_nan(tmp75, tmp27);
                        auto tmp77 = tmp27 - tmp76;
                        auto tmp78 = tmp69 ? tmp77 : tmp30;
                        auto tmp79 = tmp67 + tmp78;
                        auto tmp80 = tmp31 / tmp79;
                        auto tmp81 = tmp15 + tmp14;
                        auto tmp82 = c10::convert<long>(344);
                        auto tmp83 = min_propagate_nan(tmp81, tmp82);


                        auto tmp84 = in_ptr0[static_cast<long>(x1 + (3L*x3) + (1368L*tmp83) + (471960L*x0))];
                        auto tmp89 = in_ptr0[static_cast<long>(x1 + (3L*x3) + (1368L*tmp88) + (471960L*x0))];
                        auto tmp95 = in_ptr0[static_cast<long>(x1 + (3L*x3) + (1368L*tmp94) + (471960L*x0))];
                        auto tmp101 = in_ptr0[static_cast<long>(x1 + (3L*x3) + (1368L*tmp100) + (471960L*x0))];

                        auto tmp85 = decltype(tmp80)(tmp80 * tmp84);
                        auto tmp86 = tmp42 / tmp79;
                        auto tmp87 = tmp15 + tmp32;
                        auto tmp88 = min_propagate_nan(tmp87, tmp82);


                        auto tmp90 = decltype(tmp86)(tmp86 * tmp89);
                        auto tmp91 = tmp85 + tmp90;
                        auto tmp92 = tmp54 / tmp79;
                        auto tmp93 = tmp15 + tmp44;
                        auto tmp94 = min_propagate_nan(tmp93, tmp82);


                        auto tmp96 = decltype(tmp92)(tmp92 * tmp95);
                        auto tmp97 = tmp91 + tmp96;
                        auto tmp98 = tmp66 / tmp79;
                        auto tmp99 = tmp15 + tmp56;
                        auto tmp100 = min_propagate_nan(tmp99, tmp82);


                        auto tmp102 = decltype(tmp98)(tmp98 * tmp101);
                        auto tmp103 = tmp97 + tmp102;
                        auto tmp104 = tmp78 / tmp79;
                        auto tmp105 = tmp15 + tmp68;
                        auto tmp106 = min_propagate_nan(tmp105, tmp82);
                        auto tmp107 = in_ptr0[static_cast<long>(x1 + (3L*x3) + (1368L*tmp106) + (471960L*x0))];
                        auto tmp108 = decltype(tmp104)(tmp104 * tmp107);
                        auto tmp109 = tmp103 + tmp108;
                        out_ptr0[static_cast<long>(x3 + (456L*x2) + (123576L*x1) + (370728L*x0))] = tmp109;
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(123576L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(x2 + (123576L*x1) + (370728L*x0))];
                    out_ptr1[static_cast<long>(x1 + (3L*x2) + (370728L*x0))] = tmp0;
                }
            }
        }
    }
}
''')


cpp_fused_clone_index_mul_sum_1 = async_compile.cpp('''
#include "/tmp/torchinductor_root/bf/cbf262yqjxhzxmw7lov36xiiezas3czyjs7cdvyrvlrje4xcl2kd.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x2=static_cast<long>(0L); x2<static_cast<long>(271L); x2+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x3=static_cast<long>(0L); x3<static_cast<long>(456L); x3+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = c10::convert<float>(tmp0);
                        auto tmp2 = c10::convert<float>(0.5);
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = c10::convert<float>(1.2730627306273063);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = tmp5 + tmp4;
                        auto tmp7 = tmp6 + tmp2;
                        auto tmp8 = c10::convert<long>(tmp7);
                        auto tmp9 = c10::convert<long>(345);
                        auto tmp10 = min_propagate_nan(tmp8, tmp9);
                        auto tmp11 = tmp5 - tmp4;
                        auto tmp12 = tmp11 + tmp2;
                        auto tmp13 = c10::convert<long>(tmp12);
                        auto tmp14 = c10::convert<long>(0);
                        auto tmp15 = max_propagate_nan(tmp13, tmp14);
                        auto tmp16 = tmp10 - tmp15;
                        auto tmp17 = c10::convert<long>(5);
                        auto tmp18 = min_propagate_nan(tmp16, tmp17);
                        auto tmp19 = tmp14 < tmp18;
                        auto tmp20 = tmp14 + tmp15;
                        auto tmp21 = c10::convert<float>(tmp20);
                        auto tmp22 = tmp21 - tmp5;
                        auto tmp23 = tmp22 + tmp2;
                        auto tmp24 = c10::convert<float>(0.7855072463768116);
                        auto tmp25 = decltype(tmp23)(tmp23 * tmp24);
                        auto tmp26 = std::abs(tmp25);
                        auto tmp27 = c10::convert<float>(1.0);
                        auto tmp28 = min_propagate_nan(tmp26, tmp27);
                        auto tmp29 = tmp27 - tmp28;
                        auto tmp30 = c10::convert<float>(0.0);
                        auto tmp31 = tmp19 ? tmp29 : tmp30;
                        auto tmp32 = c10::convert<long>(1);
                        auto tmp33 = tmp32 < tmp18;
                        auto tmp34 = tmp32 + tmp15;
                        auto tmp35 = c10::convert<float>(tmp34);
                        auto tmp36 = tmp35 - tmp5;
                        auto tmp37 = tmp36 + tmp2;
                        auto tmp38 = decltype(tmp37)(tmp37 * tmp24);
                        auto tmp39 = std::abs(tmp38);
                        auto tmp40 = min_propagate_nan(tmp39, tmp27);
                        auto tmp41 = tmp27 - tmp40;
                        auto tmp42 = tmp33 ? tmp41 : tmp30;
                        auto tmp43 = tmp31 + tmp42;
                        auto tmp44 = c10::convert<long>(2);
                        auto tmp45 = tmp44 < tmp18;
                        auto tmp46 = tmp44 + tmp15;
                        auto tmp47 = c10::convert<float>(tmp46);
                        auto tmp48 = tmp47 - tmp5;
                        auto tmp49 = tmp48 + tmp2;
                        auto tmp50 = decltype(tmp49)(tmp49 * tmp24);
                        auto tmp51 = std::abs(tmp50);
                        auto tmp52 = min_propagate_nan(tmp51, tmp27);
                        auto tmp53 = tmp27 - tmp52;
                        auto tmp54 = tmp45 ? tmp53 : tmp30;
                        auto tmp55 = tmp43 + tmp54;
                        auto tmp56 = c10::convert<long>(3);
                        auto tmp57 = tmp56 < tmp18;
                        auto tmp58 = tmp56 + tmp15;
                        auto tmp59 = c10::convert<float>(tmp58);
                        auto tmp60 = tmp59 - tmp5;
                        auto tmp61 = tmp60 + tmp2;
                        auto tmp62 = decltype(tmp61)(tmp61 * tmp24);
                        auto tmp63 = std::abs(tmp62);
                        auto tmp64 = min_propagate_nan(tmp63, tmp27);
                        auto tmp65 = tmp27 - tmp64;
                        auto tmp66 = tmp57 ? tmp65 : tmp30;
                        auto tmp67 = tmp55 + tmp66;
                        auto tmp68 = c10::convert<long>(4);
                        auto tmp69 = tmp68 < tmp18;
                        auto tmp70 = tmp68 + tmp15;
                        auto tmp71 = c10::convert<float>(tmp70);
                        auto tmp72 = tmp71 - tmp5;
                        auto tmp73 = tmp72 + tmp2;
                        auto tmp74 = decltype(tmp73)(tmp73 * tmp24);
                        auto tmp75 = std::abs(tmp74);
                        auto tmp76 = min_propagate_nan(tmp75, tmp27);
                        auto tmp77 = tmp27 - tmp76;
                        auto tmp78 = tmp69 ? tmp77 : tmp30;
                        auto tmp79 = tmp67 + tmp78;
                        auto tmp80 = tmp31 / tmp79;
                                                    
                        auto tmp81 = tmp15 + tmp14;
                        auto tmp82 = c10::convert<long>(344);
                        auto tmp83 = min_propagate_nan(tmp81, tmp82);
                        auto tmp84 = in_ptr0[static_cast<long>(x1 + (3L*x3) + (1368L*tmp83) + (471960L*x0))];

                        auto tmp85 = decltype(tmp80)(tmp80 * tmp84);
                        auto tmp86 = tmp42 / tmp79;
                        auto tmp87 = tmp15 + tmp32;
                        auto tmp88 = min_propagate_nan(tmp87, tmp82);
                        auto tmp89 = in_ptr0[static_cast<long>(x1 + (3L*x3) + (1368L*tmp88) + (471960L*x0))];
                        auto tmp90 = decltype(tmp86)(tmp86 * tmp89);
                        auto tmp91 = tmp85 + tmp90;
                        auto tmp92 = tmp54 / tmp79;
                        auto tmp93 = tmp15 + tmp44;
                        auto tmp94 = min_propagate_nan(tmp93, tmp82);
                        auto tmp95 = in_ptr0[static_cast<long>(x1 + (3L*x3) + (1368L*tmp94) + (471960L*x0))];
                        auto tmp96 = decltype(tmp92)(tmp92 * tmp95);
                        auto tmp97 = tmp91 + tmp96;
                        auto tmp98 = tmp66 / tmp79;
                        auto tmp99 = tmp15 + tmp56;
                        auto tmp100 = min_propagate_nan(tmp99, tmp82);
                        auto tmp101 = in_ptr0[static_cast<long>(x1 + (3L*x3) + (1368L*tmp100) + (471960L*x0))];
                        auto tmp102 = decltype(tmp98)(tmp98 * tmp101);
                        auto tmp103 = tmp97 + tmp102;
                        auto tmp104 = tmp78 / tmp79;
                        auto tmp105 = tmp15 + tmp68;
                        auto tmp106 = min_propagate_nan(tmp105, tmp82);
                        auto tmp107 = in_ptr0[static_cast<long>(x1 + (3L*x3) + (1368L*tmp106) + (471960L*x0))];
                        auto tmp108 = decltype(tmp104)(tmp104 * tmp107);
                        auto tmp109 = tmp103 + tmp108;
                        out_ptr1[static_cast<long>(x1 + (3L*x3) + (1368L*x2) + (370728*x0))] = tmp109;
                    }
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
    assert_size_stride(arg0_1, (4, 3, 345, 456), (471960, 1, 1368, 3))
    buf3 = empty_strided((4, 3, 271, 456), (370728, 123576, 456, 1), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((4, 3, 271, 456), (370728, 1, 1368, 3), device='cpu', dtype=torch.float32)
    cpp_fused_clone_index_mul_sum_0(c_void_p(arg0_1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()))
    del arg0_1
    return (buf4, )


def call_1(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (4, 3, 345, 456), (471960, 1, 1368, 3))
    buf4 = empty_strided((4, 3, 271, 456), (370728, 1, 1368, 3), device='cpu', dtype=torch.float32)
    cpp_fused_clone_index_mul_sum_1(c_void_p(arg0_1.data_ptr()), c_void_p(buf4.data_ptr()))
    del arg0_1
    return (buf4, )


def benchmark_compiled_module(times=5000, repeat=200):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 3, 345, 456), (471960, 1, 1368, 3), device='cpu', dtype=torch.float32)

    return print_performance(lambda: call_1([arg0_1]), times=times, repeat=repeat)
    # return print_performance(lambda: call([arg0_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided
    arg0_1 = rand_strided((4, 3, 345, 456), (471960, 1, 1368, 3), device='cpu', dtype=torch.float32)

    expected = call([arg0_1, ])
    output = call_1([arg0_1, ])

    torch.testing.assert_close(expected[0], output[0])

    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
