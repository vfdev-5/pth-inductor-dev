
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


cpp_fused_affine_grid_generator_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/mq/cmqzxwuyo7ryvun3egqos5jq5ak4fue7d2jbopbqs7pgpkhdpfh4.h"
extern "C" void kernel(float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(345L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(456L); i1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i2=static_cast<long>(0L); i2<static_cast<long>(3L); i2+=static_cast<long>(1L))
                {
                    auto tmp0 = static_cast<long>(i2);
                    auto tmp1 = static_cast<long>(1);
                    auto tmp2 = tmp0 < tmp1;
                    auto tmp3 = [&]
                    {
                        auto tmp4 = static_cast<long>(i1);
                        auto tmp5 = static_cast<float>(tmp4);
                        auto tmp6 = static_cast<float>(228.0);
                        auto tmp7 = tmp5 < tmp6;
                        auto tmp8 = static_cast<float>(0.004385964912280702);
                        auto tmp9 = decltype(tmp5)(tmp5 * tmp8);
                        auto tmp10 = static_cast<float>(-0.9978070175438597);
                        auto tmp11 = tmp9 + tmp10;
                        auto tmp12 = static_cast<long>(455L + ((-1L)*i1));
                        auto tmp13 = static_cast<float>(tmp12);
                        auto tmp14 = decltype(tmp13)(tmp13 * tmp8);
                        auto tmp15 = static_cast<float>(0.9978070175438597);
                        auto tmp16 = tmp15 - tmp14;
                        auto tmp17 = tmp7 ? tmp11 : tmp16;
                        return tmp17;
                    }
                    ;
                    auto tmp18 = tmp2 ? tmp3() : static_cast<decltype(tmp3())>(0.0);
                    auto tmp19 = static_cast<long>((-1L) + i2);
                    auto tmp20 = static_cast<long>(0);
                    auto tmp21 = tmp19 >= tmp20;
                    auto tmp22 = tmp19 < tmp1;
                    auto tmp23 = tmp21 & tmp22;
                    auto tmp24 = [&]
                    {
                        auto tmp25 = static_cast<long>(i0);
                        auto tmp26 = static_cast<float>(tmp25);
                        auto tmp27 = static_cast<float>(172.5);
                        auto tmp28 = tmp26 < tmp27;
                        auto tmp29 = static_cast<float>(0.005797101449275362);
                        auto tmp30 = decltype(tmp26)(tmp26 * tmp29);
                        auto tmp31 = static_cast<float>(-0.9971014492753624);
                        auto tmp32 = tmp30 + tmp31;
                        auto tmp33 = static_cast<long>(344L + ((-1L)*i0));
                        auto tmp34 = static_cast<float>(tmp33);
                        auto tmp35 = decltype(tmp34)(tmp34 * tmp29);
                        auto tmp36 = static_cast<float>(0.9971014492753624);
                        auto tmp37 = tmp36 - tmp35;
                        auto tmp38 = tmp28 ? tmp32 : tmp37;
                        return tmp38;
                    }
                    ;
                    auto tmp39 = tmp23 ? tmp24() : static_cast<decltype(tmp24())>(0.0);
                    auto tmp40 = tmp18 + tmp39;
                    auto tmp41 = static_cast<long>((-2L) + i2);
                    auto tmp42 = tmp41 >= tmp20;
                    auto tmp43 = [&]
                    {
                        auto tmp44 = static_cast<float>(1.0);
                        return tmp44;
                    }
                    ;
                    auto tmp45 = tmp42 ? tmp43() : static_cast<decltype(tmp43())>(0.0);
                    auto tmp46 = tmp40 + tmp45;
                    out_ptr0[static_cast<long>(i2 + (3L*i1) + (1368L*i0))] = tmp46;
                }
            }
        }
    }
}
''')


cpp_fused_grid_sampler_2d_1 = async_compile.cpp('''
#include "/tmp/torchinductor_root/mq/cmqzxwuyo7ryvun3egqos5jq5ak4fue7d2jbopbqs7pgpkhdpfh4.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       const float* in_ptr1)
{
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(3L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(157320L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = in_ptr0[static_cast<long>(2L*i1)];
                auto tmp10 = in_ptr0[static_cast<long>(1L + (2L*i1))];
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
                auto tmp27 = in_ptr1[static_cast<long>(tmp26 + (456L*tmp24) + (157320L*i0))];
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
                auto tmp42 = in_ptr1[static_cast<long>(tmp41 + (456L*tmp39) + (157320L*i0))];
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
                auto tmp54 = in_ptr1[static_cast<long>(tmp53 + (456L*tmp52) + (157320L*i0))];
                auto tmp55 = tmp14 - tmp15;
                auto tmp56 = decltype(tmp30)(tmp30 * tmp55);
                auto tmp57 = tmp50 ? tmp56 : tmp6;
                auto tmp58 = tmp36 && tmp48;
                auto tmp59 = tmp35 && tmp58;
                auto tmp60 = decltype(tmp43)(tmp43 * tmp55);
                auto tmp61 = tmp59 ? tmp60 : tmp6;
                auto tmp62 = tmp59 ? tmp51 : tmp23;
                auto tmp63 = tmp59 ? tmp40 : tmp23;
                auto tmp64 = decltype(tmp27)(tmp27 * tmp34);
                auto tmp65 = decltype(tmp42)(tmp42 * tmp45);
                auto tmp66 = tmp64 + tmp65;
                auto tmp67 = decltype(tmp54)(tmp54 * tmp57);
                auto tmp68 = tmp66 + tmp67;
                auto tmp69 = in_ptr1[static_cast<long>(tmp63 + (456L*tmp62) + (157320L*i0))];
                auto tmp70 = decltype(tmp69)(tmp69 * tmp61);
                auto tmp71 = tmp68 + tmp70;
                in_out_ptr0[static_cast<long>(i1 + (157320L*i0))] = tmp71;
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
    assert_size_stride(arg0_1, (1, 3, 345, 456), (471960, 157320, 456, 1))
    assert_size_stride(arg1_1, (1, 2, 3), (6, 3, 1))
    buf0 = empty_strided((1, 345, 456, 3), (471960, 1368, 3, 1), device='cpu', dtype=torch.float32)
    cpp_fused_affine_grid_generator_0(c_void_p(buf0.data_ptr()))
    buf1 = empty_strided((1, 157320, 2), (314640, 2, 1), device='cpu', dtype=torch.float32)
    extern_kernels.bmm(as_strided(buf0, (1, 157320, 3), (0, 3, 1)), as_strided(arg1_1, (1, 3, 2), (6, 1, 3)), out=buf1)
    del arg1_1
    buf10 = as_strided(buf0, (1, 3, 345, 456), (471960, 157320, 456, 1)); del buf0  # reuse
    buf11 = buf10; del buf10  # reuse
    cpp_fused_grid_sampler_2d_1(c_void_p(buf11.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(arg0_1.data_ptr()))
    del arg0_1
    return (buf11, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 3, 345, 456), (471960, 157320, 456, 1), device='cpu', dtype=torch.float32)
    arg1_1 = rand_strided((1, 2, 3), (6, 3, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.utils import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
