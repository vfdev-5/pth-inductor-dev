
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align

from torch import device, empty, empty_strided
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
alloc_from_pool = torch.ops.inductor._alloc_from_pool
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


cpp_fused__to_copy__unsafe_index_mul_round_stack_sum_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(const unsigned char* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       unsigned char* out_ptr2,
                       unsigned char* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = c10::convert<long>(x1);
                auto tmp1 = static_cast<long>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<long>(1);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = c10::convert<long>(x0);
                    auto tmp7 = c10::convert<float>(tmp6);
                    auto tmp8 = static_cast<float>(0.5);
                    auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                    auto tmp10 = static_cast<float>(1.953125);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp12 = decltype(tmp11)(tmp11 - tmp8);
                    auto tmp13 = static_cast<float>(0.0);
                    auto tmp14 = max_propagate_nan(tmp12, tmp13);
                    auto tmp15 = c10::convert<int>(tmp14);
                    auto tmp16 = c10::convert<float>(tmp15);
                    auto tmp17 = decltype(tmp14)(tmp14 - tmp16);
                    auto tmp18 = max_propagate_nan(tmp17, tmp13);
                    auto tmp19 = static_cast<float>(1.0);
                    auto tmp20 = min_propagate_nan(tmp18, tmp19);
                    auto tmp21 = decltype(tmp19)(tmp19 - tmp20);
                    return tmp21;
                }
                ;
                auto tmp22 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                auto tmp23 = tmp0 >= tmp3;
                auto tmp24 = static_cast<long>(2);
                auto tmp25 = tmp0 < tmp24;
                auto tmp26 = [&]
                {
                    auto tmp27 = c10::convert<long>(x0);
                    auto tmp28 = c10::convert<float>(tmp27);
                    auto tmp29 = static_cast<float>(0.5);
                    auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                    auto tmp31 = static_cast<float>(1.953125);
                    auto tmp32 = decltype(tmp30)(tmp30 * tmp31);
                    auto tmp33 = decltype(tmp32)(tmp32 - tmp29);
                    auto tmp34 = static_cast<float>(0.0);
                    auto tmp35 = max_propagate_nan(tmp33, tmp34);
                    auto tmp36 = c10::convert<int>(tmp35);
                    auto tmp37 = c10::convert<float>(tmp36);
                    auto tmp38 = decltype(tmp35)(tmp35 - tmp37);
                    auto tmp39 = max_propagate_nan(tmp38, tmp34);
                    auto tmp40 = static_cast<float>(1.0);
                    auto tmp41 = min_propagate_nan(tmp39, tmp40);
                    return tmp41;
                }
                ;
                auto tmp42 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                auto tmp43 = tmp4 ? tmp22 : tmp42;
                out_ptr0[static_cast<long>(x1 + (2L*x0))] = tmp43;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(256L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(2L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = c10::convert<long>(x1);
                auto tmp1 = static_cast<long>(0);
                auto tmp2 = tmp0 >= tmp1;
                auto tmp3 = static_cast<long>(1);
                auto tmp4 = tmp0 < tmp3;
                auto tmp5 = [&]
                {
                    auto tmp6 = c10::convert<long>(x0);
                    auto tmp7 = c10::convert<float>(tmp6);
                    auto tmp8 = static_cast<float>(0.5);
                    auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                    auto tmp10 = static_cast<float>(1.953125);
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp12 = decltype(tmp11)(tmp11 - tmp8);
                    auto tmp13 = static_cast<float>(0.0);
                    auto tmp14 = max_propagate_nan(tmp12, tmp13);
                    auto tmp15 = c10::convert<int>(tmp14);
                    auto tmp16 = c10::convert<float>(tmp15);
                    auto tmp17 = decltype(tmp14)(tmp14 - tmp16);
                    auto tmp18 = max_propagate_nan(tmp17, tmp13);
                    auto tmp19 = static_cast<float>(1.0);
                    auto tmp20 = min_propagate_nan(tmp18, tmp19);
                    auto tmp21 = decltype(tmp19)(tmp19 - tmp20);
                    return tmp21;
                }
                ;
                auto tmp22 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                auto tmp23 = tmp0 >= tmp3;
                auto tmp24 = static_cast<long>(2);
                auto tmp25 = tmp0 < tmp24;
                auto tmp26 = [&]
                {
                    auto tmp27 = c10::convert<long>(x0);
                    auto tmp28 = c10::convert<float>(tmp27);
                    auto tmp29 = static_cast<float>(0.5);
                    auto tmp30 = decltype(tmp28)(tmp28 + tmp29);
                    auto tmp31 = static_cast<float>(1.953125);
                    auto tmp32 = decltype(tmp30)(tmp30 * tmp31);
                    auto tmp33 = decltype(tmp32)(tmp32 - tmp29);
                    auto tmp34 = static_cast<float>(0.0);
                    auto tmp35 = max_propagate_nan(tmp33, tmp34);
                    auto tmp36 = c10::convert<int>(tmp35);
                    auto tmp37 = c10::convert<float>(tmp36);
                    auto tmp38 = decltype(tmp35)(tmp35 - tmp37);
                    auto tmp39 = max_propagate_nan(tmp38, tmp34);
                    auto tmp40 = static_cast<float>(1.0);
                    auto tmp41 = min_propagate_nan(tmp39, tmp40);
                    return tmp41;
                }
                ;
                auto tmp42 = tmp23 ? tmp26() : static_cast<decltype(tmp26())>(0.0);
                auto tmp43 = tmp4 ? tmp22 : tmp42;
                out_ptr1[static_cast<long>(x1 + (2L*x0))] = tmp43;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(2L); x2+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(256L); x3+=static_cast<long>(1L))
                    {
                        #pragma GCC ivdep
                        for(long x4=static_cast<long>(0L); x4<static_cast<long>(2L); x4+=static_cast<long>(1L))
                        {
                            auto tmp0 = c10::convert<long>(x2);
                            auto tmp1 = static_cast<long>(0);
                            auto tmp2 = tmp0 >= tmp1;
                            auto tmp3 = static_cast<long>(1);
                            auto tmp4 = tmp0 < tmp3;
                            auto tmp5 = [&]
                            {
                                auto tmp6 = c10::convert<long>(x1);
                                auto tmp7 = c10::convert<float>(tmp6);
                                auto tmp8 = static_cast<float>(0.5);
                                auto tmp9 = decltype(tmp7)(tmp7 + tmp8);
                                auto tmp10 = static_cast<float>(1.953125);
                                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                                auto tmp12 = decltype(tmp11)(tmp11 - tmp8);
                                auto tmp13 = static_cast<float>(0.0);
                                auto tmp14 = max_propagate_nan(tmp12, tmp13);
                                auto tmp15 = c10::convert<int>(tmp14);
                                return tmp15;
                            }
                            ;
                            auto tmp16 = tmp4 ? tmp5() : static_cast<decltype(tmp5())>(0.0);
                            auto tmp17 = tmp0 >= tmp3;
                            auto tmp18 = static_cast<long>(2);
                            auto tmp19 = tmp0 < tmp18;
                            auto tmp20 = [&]
                            {
                                auto tmp21 = c10::convert<long>(x1);
                                auto tmp22 = c10::convert<float>(tmp21);
                                auto tmp23 = static_cast<float>(0.5);
                                auto tmp24 = decltype(tmp22)(tmp22 + tmp23);
                                auto tmp25 = static_cast<float>(1.953125);
                                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                                auto tmp27 = decltype(tmp26)(tmp26 - tmp23);
                                auto tmp28 = static_cast<float>(0.0);
                                auto tmp29 = max_propagate_nan(tmp27, tmp28);
                                auto tmp30 = c10::convert<int>(tmp29);
                                auto tmp31 = static_cast<int>(1);
                                auto tmp32 = decltype(tmp30)(tmp30 + tmp31);
                                return tmp32;
                            }
                            ;
                            auto tmp33 = tmp17 ? tmp20() : static_cast<decltype(tmp20())>(0.0);
                            auto tmp34 = tmp4 ? tmp16 : tmp33;
                            auto tmp35 = static_cast<int>(499);
                            auto tmp36 = min_propagate_nan(tmp34, tmp35);
                            auto tmp37 = decltype(tmp36)(tmp36 + 500);
                            auto tmp38 = tmp36 < 0;
                            auto tmp39 = tmp38 ? tmp37 : tmp36;
                            auto tmp40 = c10::convert<long>(x4);
                            auto tmp41 = tmp40 >= tmp1;
                            auto tmp42 = tmp40 < tmp3;
                            auto tmp43 = [&]
                            {
                                auto tmp44 = c10::convert<long>(x3);
                                auto tmp45 = c10::convert<float>(tmp44);
                                auto tmp46 = static_cast<float>(0.5);
                                auto tmp47 = decltype(tmp45)(tmp45 + tmp46);
                                auto tmp48 = static_cast<float>(1.953125);
                                auto tmp49 = decltype(tmp47)(tmp47 * tmp48);
                                auto tmp50 = decltype(tmp49)(tmp49 - tmp46);
                                auto tmp51 = static_cast<float>(0.0);
                                auto tmp52 = max_propagate_nan(tmp50, tmp51);
                                auto tmp53 = c10::convert<int>(tmp52);
                                return tmp53;
                            }
                            ;
                            auto tmp54 = tmp42 ? tmp43() : static_cast<decltype(tmp43())>(0.0);
                            auto tmp55 = tmp40 >= tmp3;
                            auto tmp56 = tmp40 < tmp18;
                            auto tmp57 = [&]
                            {
                                auto tmp58 = c10::convert<long>(x3);
                                auto tmp59 = c10::convert<float>(tmp58);
                                auto tmp60 = static_cast<float>(0.5);
                                auto tmp61 = decltype(tmp59)(tmp59 + tmp60);
                                auto tmp62 = static_cast<float>(1.953125);
                                auto tmp63 = decltype(tmp61)(tmp61 * tmp62);
                                auto tmp64 = decltype(tmp63)(tmp63 - tmp60);
                                auto tmp65 = static_cast<float>(0.0);
                                auto tmp66 = max_propagate_nan(tmp64, tmp65);
                                auto tmp67 = c10::convert<int>(tmp66);
                                auto tmp68 = static_cast<int>(1);
                                auto tmp69 = decltype(tmp67)(tmp67 + tmp68);
                                return tmp69;
                            }
                            ;
                            auto tmp70 = tmp55 ? tmp57() : static_cast<decltype(tmp57())>(0.0);
                            auto tmp71 = tmp42 ? tmp54 : tmp70;
                            auto tmp72 = min_propagate_nan(tmp71, tmp35);
                            auto tmp73 = decltype(tmp72)(tmp72 + 500);
                            auto tmp74 = tmp72 < 0;
                            auto tmp75 = tmp74 ? tmp73 : tmp72;
                            auto tmp76 = in_ptr0[static_cast<long>(tmp75 + (500L*tmp39) + (250000L*x0))];
                            out_ptr2[static_cast<long>(x4 + (2L*x3) + (512L*x2) + (1024L*x1) + (262144L*x0))] = tmp76;
                        }
                    }
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
            {
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(8L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(2L*x1)];
                    auto tmp1 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = out_ptr1[static_cast<long>((2L*x2) + (2L*x2_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp2 = ([&]() { __at_align__ unsigned char tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = out_ptr2[static_cast<long>((2L*x2) + (2L*x2_inner) + (1024L*x1) + (262144L*x0))]; return at::vec::Vectorized<uint8_t>::loadu_one_fourth(tmpbuf); })();
                    auto tmp7 = ([&]() { __at_align__ float tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = out_ptr1[static_cast<long>(1L + (2L*x2) + (2L*x2_inner))]; return at::vec::Vectorized<float>::loadu(tmpbuf); })();
                    auto tmp8 = ([&]() { __at_align__ unsigned char tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = out_ptr2[static_cast<long>(1L + (2L*x2) + (2L*x2_inner) + (1024L*x1) + (262144L*x0))]; return at::vec::Vectorized<uint8_t>::loadu_one_fourth(tmpbuf); })();
                    auto tmp13 = out_ptr0[static_cast<long>(1L + (2L*x1))];
                    auto tmp14 = ([&]() { __at_align__ unsigned char tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = out_ptr2[static_cast<long>(512L + (2L*x2) + (2L*x2_inner) + (1024L*x1) + (262144L*x0))]; return at::vec::Vectorized<uint8_t>::loadu_one_fourth(tmpbuf); })();
                    auto tmp20 = ([&]() { __at_align__ unsigned char tmpbuf[8]; for (long x2_inner = 0; x2_inner < 8; x2_inner++) tmpbuf[x2_inner] = out_ptr2[static_cast<long>(513L + (2L*x2) + (2L*x2_inner) + (1024L*x1) + (262144L*x0))]; return at::vec::Vectorized<uint8_t>::loadu_one_fourth(tmpbuf); })();
                    auto tmp3 = at::vec::convert_uint8_to_float(tmp2);
                    auto tmp4 = tmp1 * tmp3;
                    auto tmp5 = at::vec::Vectorized<float>(tmp0);
                    auto tmp6 = tmp5 * tmp4;
                    auto tmp9 = at::vec::convert_uint8_to_float(tmp8);
                    auto tmp10 = tmp7 * tmp9;
                    auto tmp11 = tmp5 * tmp10;
                    auto tmp12 = tmp6 + tmp11;
                    auto tmp15 = at::vec::convert_uint8_to_float(tmp14);
                    auto tmp16 = tmp1 * tmp15;
                    auto tmp17 = at::vec::Vectorized<float>(tmp13);
                    auto tmp18 = tmp17 * tmp16;
                    auto tmp19 = tmp12 + tmp18;
                    auto tmp21 = at::vec::convert_uint8_to_float(tmp20);
                    auto tmp22 = tmp7 * tmp21;
                    auto tmp23 = tmp17 * tmp22;
                    auto tmp24 = tmp19 + tmp23;
                    auto tmp25 = tmp24.round();
                    auto tmp26 = at::vec::convert_float_to_uint8(tmp25);
                    tmp26.store(out_ptr3 + static_cast<long>(x2 + (256L*x1) + (65536L*x0)), 8);
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
    assert_size_stride(arg0_1, (2, 3, 500, 500), (750000, 250000, 500, 1))
    buf0 = empty((256, 2), device='cpu', dtype=torch.float32)
    buf1 = empty((256, 2), device='cpu', dtype=torch.float32)
    buf2 = empty((2, 3, 256, 2, 256, 2), device='cpu', dtype=torch.uint8)
    buf3 = empty((2, 3, 256, 256), device='cpu', dtype=torch.uint8)
    cpp_fused__to_copy__unsafe_index_mul_round_stack_sum_0(c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()))
    del arg0_1
    return (buf3, )


def benchmark_compiled_module(times=10000, repeat=500):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 3, 500, 500), (750000, 250000, 500, 1), device='cpu', dtype=torch.uint8)
    return print_performance(lambda: call([arg0_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
