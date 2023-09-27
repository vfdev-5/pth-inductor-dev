
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


cpp_fused_affine_grid_generator_grid_sampler_2d_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/i5/ci5uspp363v3ky6jkccllm3bxudy2fkdpqinkqhmpehfihejs7ko.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       float* out_ptr4,
                       float* out_ptr5,
                       float* out_ptr6,
                       long* out_ptr7,
                       long* out_ptr8)
{
    auto out_ptr9 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(8L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(157320L); i1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i2=static_cast<long>(0L); i2<static_cast<long>(2L); i2+=static_cast<long>(1L))
                {
                    auto tmp46 = in_ptr0[static_cast<long>((3L*i2) + (6L*i0))];
                    auto tmp88 = in_ptr0[static_cast<long>(1L + (3L*i2) + (6L*i0))];
                    auto tmp132 = in_ptr0[static_cast<long>(2L + (3L*i2) + (6L*i0))];
                    auto tmp0 = static_cast<long>(0);
                    auto tmp1 = static_cast<long>(1);
                    auto tmp2 = tmp0 < tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = static_cast<long>(static_cast<long>(i1) % static_cast<long>(456L));
                        auto tmp5 = static_cast<float>(tmp4);
                        auto tmp6 = static_cast<float>(228.0);
                        auto tmp7 = tmp5 < tmp6;
                        auto tmp8 = static_cast<float>(0.004385964912280702);
                        auto tmp9 = decltype(tmp5)(tmp5 * tmp8);
                        auto tmp10 = static_cast<float>(-0.9978070175438597);
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = static_cast<long>(455L + ((-1L)*(static_cast<long>(i1) % static_cast<long>(456L))));
                        auto tmp13 = static_cast<float>(tmp12);
                        auto tmp14 = decltype(tmp13)(tmp13 * tmp8);
                        auto tmp15 = static_cast<float>(0.9978070175438597);
                        auto tmp16 = tmp15 - tmp14;
                        auto tmp17 = tmp7 ? tmp11 : tmp16;
                        return tmp17;
                    }
                    ;
                    auto tmp18 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                    auto tmp19 = static_cast<long>(-1);
                    auto tmp20 = tmp19 >= tmp0;
                    auto tmp21 = tmp19 < tmp1;
                    auto tmp22 = tmp20 & tmp21;
                    auto tmp23 = [&]
                    {
                        auto tmp24 = static_cast<long>(at::native::div_floor_integer(i1, 456L));
                        auto tmp25 = static_cast<float>(tmp24);
                        auto tmp26 = static_cast<float>(172.5);
                        auto tmp27 = tmp25 < tmp26;
                        auto tmp28 = static_cast<float>(0.005797101449275362);
                        auto tmp29 = decltype(tmp25)(tmp25 * tmp28);
                        auto tmp30 = static_cast<float>(-0.9971014492753624);
                        auto tmp31 = tmp29 + tmp30;
                        auto tmp32 = static_cast<long>(344L + ((-1L)*(at::native::div_floor_integer(i1, 456L))));
                        auto tmp33 = static_cast<float>(tmp32);
                        auto tmp34 = decltype(tmp33)(tmp33 * tmp28);
                        auto tmp35 = static_cast<float>(0.9971014492753624);
                        auto tmp36 = tmp35 - tmp34;
                        auto tmp37 = tmp27 ? tmp31 : tmp36;
                        return tmp37;
                    }
                    ;
                    auto tmp38 = tmp22 ? tmp23() : static_cast<decltype(tmp23())>(0.0);
                    auto tmp39 = tmp18 + tmp38;
                    auto tmp40 = static_cast<long>(-2);
                    auto tmp41 = tmp40 >= tmp0;
                    auto tmp42 = [&]
                    {
                        auto tmp43 = static_cast<float>(1.0);
                        return tmp43;
                    }
                    ;
                    auto tmp44 = tmp41 ? tmp42() : static_cast<decltype(tmp42())>(0.0);
                    auto tmp45 = tmp39 + tmp44;
                    auto tmp47 = decltype(tmp45)(tmp45 * tmp46);
                    auto tmp48 = tmp1 < tmp1;
                    auto tmp49 = [&]
                    {
                        auto tmp50 = static_cast<long>(static_cast<long>(i1) % static_cast<long>(456L));
                        auto tmp51 = static_cast<float>(tmp50);
                        auto tmp52 = static_cast<float>(228.0);
                        auto tmp53 = tmp51 < tmp52;
                        auto tmp54 = static_cast<float>(0.004385964912280702);
                        auto tmp55 = decltype(tmp51)(tmp51 * tmp54);
                        auto tmp56 = static_cast<float>(-0.9978070175438597);
                        auto tmp57 = tmp55 + tmp56;
                        auto tmp58 = static_cast<long>(455L + ((-1L)*(static_cast<long>(i1) % static_cast<long>(456L))));
                        auto tmp59 = static_cast<float>(tmp58);
                        auto tmp60 = decltype(tmp59)(tmp59 * tmp54);
                        auto tmp61 = static_cast<float>(0.9978070175438597);
                        auto tmp62 = tmp61 - tmp60;
                        auto tmp63 = tmp53 ? tmp57 : tmp62;
                        return tmp63;
                    }
                    ;
                    auto tmp64 = tmp48 ? tmp49() : static_cast<decltype(tmp49())>(0.0);
                    auto tmp65 = tmp0 >= tmp0;
                    auto tmp66 = tmp65 & tmp2;
                    auto tmp67 = [&]
                    {
                        auto tmp68 = static_cast<long>(at::native::div_floor_integer(i1, 456L));
                        auto tmp69 = static_cast<float>(tmp68);
                        auto tmp70 = static_cast<float>(172.5);
                        auto tmp71 = tmp69 < tmp70;
                        auto tmp72 = static_cast<float>(0.005797101449275362);
                        auto tmp73 = decltype(tmp69)(tmp69 * tmp72);
                        auto tmp74 = static_cast<float>(-0.9971014492753624);
                        auto tmp75 = tmp73 + tmp74;
                        auto tmp76 = static_cast<long>(344L + ((-1L)*(at::native::div_floor_integer(i1, 456L))));
                        auto tmp77 = static_cast<float>(tmp76);
                        auto tmp78 = decltype(tmp77)(tmp77 * tmp72);
                        auto tmp79 = static_cast<float>(0.9971014492753624);
                        auto tmp80 = tmp79 - tmp78;
                        auto tmp81 = tmp71 ? tmp75 : tmp80;
                        return tmp81;
                    }
                    ;
                    auto tmp82 = tmp66 ? tmp67() : static_cast<decltype(tmp67())>(0.0);
                    auto tmp83 = tmp64 + tmp82;
                    auto tmp84 = [&]
                    {
                        auto tmp85 = static_cast<float>(1.0);
                        return tmp85;
                    }
                    ;
                    auto tmp86 = tmp20 ? tmp84() : static_cast<decltype(tmp84())>(0.0);
                    auto tmp87 = tmp83 + tmp86;
                    auto tmp89 = decltype(tmp87)(tmp87 * tmp88);
                    auto tmp90 = tmp47 + tmp89;
                    auto tmp91 = static_cast<long>(2);
                    auto tmp92 = tmp91 < tmp1;
                    auto tmp93 = [&]
                    {
                        auto tmp94 = static_cast<long>(static_cast<long>(i1) % static_cast<long>(456L));
                        auto tmp95 = static_cast<float>(tmp94);
                        auto tmp96 = static_cast<float>(228.0);
                        auto tmp97 = tmp95 < tmp96;
                        auto tmp98 = static_cast<float>(0.004385964912280702);
                        auto tmp99 = decltype(tmp95)(tmp95 * tmp98);
                        auto tmp100 = static_cast<float>(-0.9978070175438597);
                        auto tmp101 = tmp99 + tmp100;
                        auto tmp102 = static_cast<long>(455L + ((-1L)*(static_cast<long>(i1) % static_cast<long>(456L))));
                        auto tmp103 = static_cast<float>(tmp102);
                        auto tmp104 = decltype(tmp103)(tmp103 * tmp98);
                        auto tmp105 = static_cast<float>(0.9978070175438597);
                        auto tmp106 = tmp105 - tmp104;
                        auto tmp107 = tmp97 ? tmp101 : tmp106;
                        return tmp107;
                    }
                    ;
                    auto tmp108 = tmp92 ? tmp93() : static_cast<decltype(tmp93())>(0.0);
                    auto tmp109 = tmp1 >= tmp0;
                    auto tmp110 = tmp109 & tmp48;
                    auto tmp111 = [&]
                    {
                        auto tmp112 = static_cast<long>(at::native::div_floor_integer(i1, 456L));
                        auto tmp113 = static_cast<float>(tmp112);
                        auto tmp114 = static_cast<float>(172.5);
                        auto tmp115 = tmp113 < tmp114;
                        auto tmp116 = static_cast<float>(0.005797101449275362);
                        auto tmp117 = decltype(tmp113)(tmp113 * tmp116);
                        auto tmp118 = static_cast<float>(-0.9971014492753624);
                        auto tmp119 = tmp117 + tmp118;
                        auto tmp120 = static_cast<long>(344L + ((-1L)*(at::native::div_floor_integer(i1, 456L))));
                        auto tmp121 = static_cast<float>(tmp120);
                        auto tmp122 = decltype(tmp121)(tmp121 * tmp116);
                        auto tmp123 = static_cast<float>(0.9971014492753624);
                        auto tmp124 = tmp123 - tmp122;
                        auto tmp125 = tmp115 ? tmp119 : tmp124;
                        return tmp125;
                    }
                    ;
                    auto tmp126 = tmp110 ? tmp111() : static_cast<decltype(tmp111())>(0.0);
                    auto tmp127 = tmp108 + tmp126;
                    auto tmp128 = [&]
                    {
                        auto tmp129 = static_cast<float>(1.0);
                        return tmp129;
                    }
                    ;
                    auto tmp130 = tmp65 ? tmp128() : static_cast<decltype(tmp128())>(0.0);
                    auto tmp131 = tmp127 + tmp130;
                    auto tmp133 = decltype(tmp131)(tmp131 * tmp132);
                    auto tmp134 = tmp90 + tmp133;
                    out_ptr0[static_cast<long>(i2 + (2L*i1) + (314640L*i0))] = tmp134;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(8L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(3L); i1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i2=static_cast<long>(0L); i2<static_cast<long>(157320L); i2+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>((2L*i2) + (314640L*i0))];
                    auto tmp10 = out_ptr0[static_cast<long>(1L + (2L*i2) + (314640L*i0))];
                    auto tmp1 = static_cast<float>(228.0);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp3 = static_cast<float>(227.5);
                    auto tmp4 = tmp2 + tmp3;
                    auto tmp5 = std::floor(tmp4);
                    auto tmp6 = static_cast<float>(0.0);
                    auto tmp7 = tmp5 >= tmp6;
                    auto tmp8 = static_cast<float>(456.0);
                    auto tmp9 = tmp5 < tmp8;
                    auto tmp11 = static_cast<float>(172.5);
                    auto tmp12 = decltype(tmp10)(tmp10 * tmp11);
                    auto tmp13 = static_cast<float>(172.0);
                    auto tmp14 = tmp12 + tmp13;
                    auto tmp15 = std::floor(tmp14);
                    auto tmp16 = tmp15 >= tmp6;
                    auto tmp17 = static_cast<float>(345.0);
                    auto tmp18 = tmp15 < tmp17;
                    auto tmp19 = tmp16 && tmp18;
                    auto tmp20 = tmp9 && tmp19;
                    auto tmp21 = tmp7 && tmp20;
                    auto tmp22 = static_cast<long>(tmp15);
                    auto tmp23 = static_cast<long>(0);
                    auto tmp24 = tmp21 ? tmp22 : tmp23;
                    auto tmp25 = static_cast<long>(tmp5);
                    auto tmp26 = tmp21 ? tmp25 : tmp23;
                    auto tmp27 = in_ptr1[static_cast<long>(tmp26 + (456L*tmp24) + (157320L*i1) + (471960L*i0))];
                    auto tmp28 = static_cast<float>(1.0);
                    auto tmp29 = tmp5 + tmp28;
                    auto tmp30 = tmp29 - tmp4;
                    auto tmp31 = tmp15 + tmp28;
                    auto tmp32 = tmp31 - tmp14;
                    auto tmp33 = decltype(tmp30)(tmp30 * tmp32);
                    auto tmp34 = tmp21 ? tmp33 : tmp6;
                    auto tmp35 = tmp29 >= tmp6;
                    auto tmp36 = tmp29 < tmp8;
                    auto tmp37 = tmp36 && tmp19;
                    auto tmp38 = tmp35 && tmp37;
                    auto tmp39 = tmp38 ? tmp22 : tmp23;
                    auto tmp40 = static_cast<long>(tmp29);
                    auto tmp41 = tmp38 ? tmp40 : tmp23;
                    auto tmp42 = in_ptr1[static_cast<long>(tmp41 + (456L*tmp39) + (157320L*i1) + (471960L*i0))];
                    auto tmp43 = tmp4 - tmp5;
                    auto tmp44 = decltype(tmp43)(tmp43 * tmp32);
                    auto tmp45 = tmp38 ? tmp44 : tmp6;
                    auto tmp46 = tmp31 >= tmp6;
                    auto tmp47 = tmp31 < tmp17;
                    auto tmp48 = tmp46 && tmp47;
                    auto tmp49 = tmp9 && tmp48;
                    auto tmp50 = tmp7 && tmp49;
                    auto tmp51 = static_cast<long>(tmp31);
                    auto tmp52 = tmp50 ? tmp51 : tmp23;
                    auto tmp53 = tmp50 ? tmp25 : tmp23;
                    auto tmp54 = in_ptr1[static_cast<long>(tmp53 + (456L*tmp52) + (157320L*i1) + (471960L*i0))];
                    auto tmp55 = tmp14 - tmp15;
                    auto tmp56 = decltype(tmp30)(tmp30 * tmp55);
                    auto tmp57 = tmp50 ? tmp56 : tmp6;
                    auto tmp58 = tmp36 && tmp48;
                    auto tmp59 = tmp35 && tmp58;
                    auto tmp60 = tmp59 ? tmp51 : tmp23;
                    auto tmp61 = tmp59 ? tmp40 : tmp23;
                    auto tmp62 = decltype(tmp43)(tmp43 * tmp55);
                    auto tmp63 = tmp59 ? tmp62 : tmp6;
                    out_ptr1[static_cast<long>(i2 + (157320L*i1) + (471960L*i0))] = tmp27;
                    out_ptr2[static_cast<long>(i2 + (157320L*i1) + (471960L*i0))] = tmp34;
                    out_ptr3[static_cast<long>(i2 + (157320L*i1) + (471960L*i0))] = tmp42;
                    out_ptr4[static_cast<long>(i2 + (157320L*i1) + (471960L*i0))] = tmp45;
                    out_ptr5[static_cast<long>(i2 + (157320L*i1) + (471960L*i0))] = tmp54;
                    out_ptr6[static_cast<long>(i2 + (157320L*i1) + (471960L*i0))] = tmp57;
                    out_ptr7[static_cast<long>(i2 + (157320L*i1) + (471960L*i0))] = tmp60;
                    out_ptr8[static_cast<long>(i2 + (157320L*i1) + (471960L*i0))] = tmp61;
                    out_ptr9[static_cast<long>(i2 + (157320L*i1) + (471960L*i0))] = tmp63;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(24L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(157320L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr1[static_cast<long>(i1 + (157320L*i0))];
                auto tmp1 = out_ptr2[static_cast<long>(i1 + (157320L*i0))];
                auto tmp3 = out_ptr3[static_cast<long>(i1 + (157320L*i0))];
                auto tmp4 = out_ptr4[static_cast<long>(i1 + (157320L*i0))];
                auto tmp7 = out_ptr5[static_cast<long>(i1 + (157320L*i0))];
                auto tmp8 = out_ptr6[static_cast<long>(i1 + (157320L*i0))];
                auto tmp11 = out_ptr7[static_cast<long>(i1 + (157320L*i0))];
                auto tmp12 = out_ptr8[static_cast<long>(i1 + (157320L*i0))];
                auto tmp14 = out_ptr9[static_cast<long>(i1 + (157320L*i0))];
                auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                auto tmp6 = tmp2 + tmp5;
                auto tmp9 = decltype(tmp7)(tmp7 * tmp8);
                auto tmp10 = tmp6 + tmp9;
                auto tmp13 = in_ptr1[static_cast<long>(tmp12 + (456L*tmp11) + (157320L*i0))];
                auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                auto tmp16 = tmp10 + tmp15;
                in_out_ptr0[static_cast<long>(i1 + (157320L*i0))] = tmp16;
            }
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, 3, 345, 456), (471960, 157320, 456, 1))
    assert_size_stride(arg1_1, (8, 2, 3), (6, 3, 1))
    buf1 = empty_strided((8, 157320, 2), (314640, 2, 1), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((8, 3, 345, 456), (471960, 157320, 456, 1), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((8, 3, 345, 456), (471960, 157320, 456, 1), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((8, 3, 345, 456), (471960, 157320, 456, 1), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((8, 3, 345, 456), (471960, 157320, 456, 1), device='cpu', dtype=torch.float32)
    buf6 = empty_strided((8, 3, 345, 456), (471960, 157320, 456, 1), device='cpu', dtype=torch.float32)
    buf7 = empty_strided((8, 3, 345, 456), (471960, 157320, 456, 1), device='cpu', dtype=torch.float32)
    buf8 = empty_strided((8, 3, 345, 456), (471960, 157320, 456, 1), device='cpu', dtype=torch.int64)
    buf9 = empty_strided((8, 3, 345, 456), (471960, 157320, 456, 1), device='cpu', dtype=torch.int64)
    buf10 = empty_strided((8, 3, 345, 456), (471960, 157320, 456, 1), device='cpu', dtype=torch.float32)
    buf11 = buf10; del buf10  # reuse
    cpp_fused_affine_grid_generator_grid_sampler_2d_0(c_void_p(buf11.data_ptr()), c_void_p(arg1_1.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf5.data_ptr()), c_void_p(buf6.data_ptr()), c_void_p(buf7.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf9.data_ptr()))
    del arg0_1
    del arg1_1
    return (buf11, )


def benchmark_compiled_module(times=1000, repeat=200):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, 3, 345, 456), (471960, 157320, 456, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((8, 2, 3), (6, 3, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
