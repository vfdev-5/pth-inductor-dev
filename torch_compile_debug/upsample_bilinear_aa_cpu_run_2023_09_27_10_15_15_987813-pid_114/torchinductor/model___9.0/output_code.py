
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


cpp_fused_add_div_index_mul_sum_unbind_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr4)
{
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(270L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = static_cast<long>(i0);
            auto tmp1 = static_cast<float>(tmp0);
            auto tmp2 = static_cast<float>(0.5);
            auto tmp3 = tmp1 + tmp2;
            auto tmp4 = static_cast<float>(1.6888888888888889);
            auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
            auto tmp6 = static_cast<float>(1.6888889074325562);
            auto tmp7 = tmp5 + tmp6;
            auto tmp8 = tmp7 + tmp2;
            auto tmp9 = static_cast<long>(tmp8);
            auto tmp10 = static_cast<long>(456);
            auto tmp11 = min_propagate_nan(tmp9, tmp10);
            auto tmp12 = tmp5 - tmp6;
            auto tmp13 = tmp12 + tmp2;
            auto tmp14 = static_cast<long>(tmp13);
            auto tmp15 = static_cast<long>(0);
            auto tmp16 = max_propagate_nan(tmp14, tmp15);
            auto tmp17 = tmp11 - tmp16;
            auto tmp18 = max_propagate_nan(tmp17, tmp15);
            auto tmp19 = static_cast<long>(5);
            auto tmp20 = min_propagate_nan(tmp18, tmp19);
            auto tmp21 = tmp15 < tmp20;
            auto tmp22 = tmp15 + tmp16;
            auto tmp23 = static_cast<float>(tmp22);
            auto tmp24 = tmp23 - tmp5;
            auto tmp25 = tmp24 + tmp2;
            auto tmp26 = static_cast<float>(0.5921052631578947);
            auto tmp27 = decltype(tmp25)(tmp25 * tmp26);
            auto tmp28 = std::abs(tmp27);
            auto tmp29 = static_cast<float>(1.0);
            auto tmp30 = min_propagate_nan(tmp28, tmp29);
            auto tmp31 = tmp29 - tmp30;
            auto tmp32 = static_cast<float>(0.0);
            auto tmp33 = tmp21 ? tmp31 : tmp32;
            auto tmp34 = static_cast<long>(1);
            auto tmp35 = tmp34 < tmp20;
            auto tmp36 = tmp34 + tmp16;
            auto tmp37 = static_cast<float>(tmp36);
            auto tmp38 = tmp37 - tmp5;
            auto tmp39 = tmp38 + tmp2;
            auto tmp40 = decltype(tmp39)(tmp39 * tmp26);
            auto tmp41 = std::abs(tmp40);
            auto tmp42 = min_propagate_nan(tmp41, tmp29);
            auto tmp43 = tmp29 - tmp42;
            auto tmp44 = tmp35 ? tmp43 : tmp32;
            auto tmp45 = tmp33 + tmp44;
            auto tmp46 = static_cast<long>(2);
            auto tmp47 = tmp46 < tmp20;
            auto tmp48 = tmp46 + tmp16;
            auto tmp49 = static_cast<float>(tmp48);
            auto tmp50 = tmp49 - tmp5;
            auto tmp51 = tmp50 + tmp2;
            auto tmp52 = decltype(tmp51)(tmp51 * tmp26);
            auto tmp53 = std::abs(tmp52);
            auto tmp54 = min_propagate_nan(tmp53, tmp29);
            auto tmp55 = tmp29 - tmp54;
            auto tmp56 = tmp47 ? tmp55 : tmp32;
            auto tmp57 = tmp45 + tmp56;
            auto tmp58 = static_cast<long>(3);
            auto tmp59 = tmp58 < tmp20;
            auto tmp60 = tmp58 + tmp16;
            auto tmp61 = static_cast<float>(tmp60);
            auto tmp62 = tmp61 - tmp5;
            auto tmp63 = tmp62 + tmp2;
            auto tmp64 = decltype(tmp63)(tmp63 * tmp26);
            auto tmp65 = std::abs(tmp64);
            auto tmp66 = min_propagate_nan(tmp65, tmp29);
            auto tmp67 = tmp29 - tmp66;
            auto tmp68 = tmp59 ? tmp67 : tmp32;
            auto tmp69 = tmp57 + tmp68;
            auto tmp70 = static_cast<long>(4);
            auto tmp71 = tmp70 < tmp20;
            auto tmp72 = tmp70 + tmp16;
            auto tmp73 = static_cast<float>(tmp72);
            auto tmp74 = tmp73 - tmp5;
            auto tmp75 = tmp74 + tmp2;
            auto tmp76 = decltype(tmp75)(tmp75 * tmp26);
            auto tmp77 = std::abs(tmp76);
            auto tmp78 = min_propagate_nan(tmp77, tmp29);
            auto tmp79 = tmp29 - tmp78;
            auto tmp80 = tmp71 ? tmp79 : tmp32;
            auto tmp81 = tmp69 + tmp80;
            out_ptr0[static_cast<long>(i0)] = tmp81;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(270L); i0+=static_cast<long>(1L))
        {
            auto tmp0 = static_cast<long>(i0);
            auto tmp1 = static_cast<float>(tmp0);
            auto tmp2 = static_cast<float>(0.5);
            auto tmp3 = tmp1 + tmp2;
            auto tmp4 = static_cast<float>(1.2777777777777777);
            auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
            auto tmp6 = static_cast<float>(1.2777777910232544);
            auto tmp7 = tmp5 + tmp6;
            auto tmp8 = tmp7 + tmp2;
            auto tmp9 = static_cast<long>(tmp8);
            auto tmp10 = static_cast<long>(345);
            auto tmp11 = min_propagate_nan(tmp9, tmp10);
            auto tmp12 = tmp5 - tmp6;
            auto tmp13 = tmp12 + tmp2;
            auto tmp14 = static_cast<long>(tmp13);
            auto tmp15 = static_cast<long>(0);
            auto tmp16 = max_propagate_nan(tmp14, tmp15);
            auto tmp17 = tmp11 - tmp16;
            auto tmp18 = max_propagate_nan(tmp17, tmp15);
            auto tmp19 = static_cast<long>(5);
            auto tmp20 = min_propagate_nan(tmp18, tmp19);
            auto tmp21 = tmp15 < tmp20;
            auto tmp22 = tmp15 + tmp16;
            auto tmp23 = static_cast<float>(tmp22);
            auto tmp24 = tmp23 - tmp5;
            auto tmp25 = tmp24 + tmp2;
            auto tmp26 = static_cast<float>(0.782608695652174);
            auto tmp27 = decltype(tmp25)(tmp25 * tmp26);
            auto tmp28 = std::abs(tmp27);
            auto tmp29 = static_cast<float>(1.0);
            auto tmp30 = min_propagate_nan(tmp28, tmp29);
            auto tmp31 = tmp29 - tmp30;
            auto tmp32 = static_cast<float>(0.0);
            auto tmp33 = tmp21 ? tmp31 : tmp32;
            auto tmp34 = static_cast<long>(1);
            auto tmp35 = tmp34 < tmp20;
            auto tmp36 = tmp34 + tmp16;
            auto tmp37 = static_cast<float>(tmp36);
            auto tmp38 = tmp37 - tmp5;
            auto tmp39 = tmp38 + tmp2;
            auto tmp40 = decltype(tmp39)(tmp39 * tmp26);
            auto tmp41 = std::abs(tmp40);
            auto tmp42 = min_propagate_nan(tmp41, tmp29);
            auto tmp43 = tmp29 - tmp42;
            auto tmp44 = tmp35 ? tmp43 : tmp32;
            auto tmp45 = tmp33 + tmp44;
            auto tmp46 = static_cast<long>(2);
            auto tmp47 = tmp46 < tmp20;
            auto tmp48 = tmp46 + tmp16;
            auto tmp49 = static_cast<float>(tmp48);
            auto tmp50 = tmp49 - tmp5;
            auto tmp51 = tmp50 + tmp2;
            auto tmp52 = decltype(tmp51)(tmp51 * tmp26);
            auto tmp53 = std::abs(tmp52);
            auto tmp54 = min_propagate_nan(tmp53, tmp29);
            auto tmp55 = tmp29 - tmp54;
            auto tmp56 = tmp47 ? tmp55 : tmp32;
            auto tmp57 = tmp45 + tmp56;
            auto tmp58 = static_cast<long>(3);
            auto tmp59 = tmp58 < tmp20;
            auto tmp60 = tmp58 + tmp16;
            auto tmp61 = static_cast<float>(tmp60);
            auto tmp62 = tmp61 - tmp5;
            auto tmp63 = tmp62 + tmp2;
            auto tmp64 = decltype(tmp63)(tmp63 * tmp26);
            auto tmp65 = std::abs(tmp64);
            auto tmp66 = min_propagate_nan(tmp65, tmp29);
            auto tmp67 = tmp29 - tmp66;
            auto tmp68 = tmp59 ? tmp67 : tmp32;
            auto tmp69 = tmp57 + tmp68;
            auto tmp70 = static_cast<long>(4);
            auto tmp71 = tmp70 < tmp20;
            auto tmp72 = tmp70 + tmp16;
            auto tmp73 = static_cast<float>(tmp72);
            auto tmp74 = tmp73 - tmp5;
            auto tmp75 = tmp74 + tmp2;
            auto tmp76 = decltype(tmp75)(tmp75 * tmp26);
            auto tmp77 = std::abs(tmp76);
            auto tmp78 = min_propagate_nan(tmp77, tmp29);
            auto tmp79 = tmp29 - tmp78;
            auto tmp80 = tmp71 ? tmp79 : tmp32;
            auto tmp81 = tmp69 + tmp80;
            out_ptr1[static_cast<long>(i0)] = tmp81;
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(5L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(270L); i1+=static_cast<long>(1L))
            {
                auto tmp35 = out_ptr0[static_cast<long>(i1)];
                auto tmp0 = static_cast<long>(i1);
                auto tmp1 = static_cast<float>(tmp0);
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = tmp1 + tmp2;
                auto tmp4 = static_cast<float>(1.6888888888888889);
                auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                auto tmp6 = static_cast<float>(1.6888889074325562);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp7 + tmp2;
                auto tmp9 = static_cast<long>(tmp8);
                auto tmp10 = static_cast<long>(456);
                auto tmp11 = min_propagate_nan(tmp9, tmp10);
                auto tmp12 = tmp5 - tmp6;
                auto tmp13 = tmp12 + tmp2;
                auto tmp14 = static_cast<long>(tmp13);
                auto tmp15 = static_cast<long>(0);
                auto tmp16 = max_propagate_nan(tmp14, tmp15);
                auto tmp17 = tmp11 - tmp16;
                auto tmp18 = max_propagate_nan(tmp17, tmp15);
                auto tmp19 = static_cast<long>(5);
                auto tmp20 = min_propagate_nan(tmp18, tmp19);
                auto tmp21 = static_cast<long>(i0);
                auto tmp22 = tmp21 < tmp20;
                auto tmp23 = tmp21 + tmp16;
                auto tmp24 = static_cast<float>(tmp23);
                auto tmp25 = tmp24 - tmp5;
                auto tmp26 = tmp25 + tmp2;
                auto tmp27 = static_cast<float>(0.5921052631578947);
                auto tmp28 = decltype(tmp26)(tmp26 * tmp27);
                auto tmp29 = std::abs(tmp28);
                auto tmp30 = static_cast<float>(1.0);
                auto tmp31 = min_propagate_nan(tmp29, tmp30);
                auto tmp32 = tmp30 - tmp31;
                auto tmp33 = static_cast<float>(0.0);
                auto tmp34 = tmp22 ? tmp32 : tmp33;
                auto tmp36 = tmp34 / tmp35;
                out_ptr2[static_cast<long>(i1 + (270L*i0))] = tmp36;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(2070L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(270L); i1+=static_cast<long>(1L))
            {
                auto tmp16 = out_ptr2[static_cast<long>(i1)];
                auto tmp22 = out_ptr2[static_cast<long>(270L + i1)];
                auto tmp29 = out_ptr2[static_cast<long>(540L + i1)];
                auto tmp36 = out_ptr2[static_cast<long>(810L + i1)];
                auto tmp0 = static_cast<long>(i1);
                auto tmp1 = static_cast<float>(tmp0);
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = tmp1 + tmp2;
                auto tmp4 = static_cast<float>(1.6888888888888889);
                auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                auto tmp6 = static_cast<float>(1.6888889074325562);
                auto tmp7 = tmp5 - tmp6;
                auto tmp8 = tmp7 + tmp2;
                auto tmp9 = static_cast<long>(tmp8);
                auto tmp10 = static_cast<long>(0);
                auto tmp11 = max_propagate_nan(tmp9, tmp10);
                auto tmp12 = tmp11 + tmp10;
                auto tmp13 = static_cast<long>(455);
                auto tmp14 = min_propagate_nan(tmp12, tmp13);
                auto tmp15 = in_ptr0[static_cast<long>(tmp14 + (456L*i0))];
                auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                auto tmp18 = static_cast<long>(1);
                auto tmp19 = tmp11 + tmp18;
                auto tmp20 = min_propagate_nan(tmp19, tmp13);
                auto tmp21 = in_ptr0[static_cast<long>(tmp20 + (456L*i0))];
                auto tmp23 = decltype(tmp21)(tmp21 * tmp22);
                auto tmp24 = tmp17 + tmp23;
                auto tmp25 = static_cast<long>(2);
                auto tmp26 = tmp11 + tmp25;
                auto tmp27 = min_propagate_nan(tmp26, tmp13);
                auto tmp28 = in_ptr0[static_cast<long>(tmp27 + (456L*i0))];
                auto tmp30 = decltype(tmp28)(tmp28 * tmp29);
                auto tmp31 = tmp24 + tmp30;
                auto tmp32 = static_cast<long>(3);
                auto tmp33 = tmp11 + tmp32;
                auto tmp34 = min_propagate_nan(tmp33, tmp13);
                auto tmp35 = in_ptr0[static_cast<long>(tmp34 + (456L*i0))];
                auto tmp37 = decltype(tmp35)(tmp35 * tmp36);
                auto tmp38 = tmp31 + tmp37;
                in_out_ptr0[static_cast<long>(i1 + (270L*i0))] = tmp38;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(5L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(270L); i1+=static_cast<long>(1L))
            {
                auto tmp35 = out_ptr1[static_cast<long>(i1)];
                auto tmp0 = static_cast<long>(i1);
                auto tmp1 = static_cast<float>(tmp0);
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = tmp1 + tmp2;
                auto tmp4 = static_cast<float>(1.2777777777777777);
                auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                auto tmp6 = static_cast<float>(1.2777777910232544);
                auto tmp7 = tmp5 + tmp6;
                auto tmp8 = tmp7 + tmp2;
                auto tmp9 = static_cast<long>(tmp8);
                auto tmp10 = static_cast<long>(345);
                auto tmp11 = min_propagate_nan(tmp9, tmp10);
                auto tmp12 = tmp5 - tmp6;
                auto tmp13 = tmp12 + tmp2;
                auto tmp14 = static_cast<long>(tmp13);
                auto tmp15 = static_cast<long>(0);
                auto tmp16 = max_propagate_nan(tmp14, tmp15);
                auto tmp17 = tmp11 - tmp16;
                auto tmp18 = max_propagate_nan(tmp17, tmp15);
                auto tmp19 = static_cast<long>(5);
                auto tmp20 = min_propagate_nan(tmp18, tmp19);
                auto tmp21 = static_cast<long>(i0);
                auto tmp22 = tmp21 < tmp20;
                auto tmp23 = tmp21 + tmp16;
                auto tmp24 = static_cast<float>(tmp23);
                auto tmp25 = tmp24 - tmp5;
                auto tmp26 = tmp25 + tmp2;
                auto tmp27 = static_cast<float>(0.782608695652174);
                auto tmp28 = decltype(tmp26)(tmp26 * tmp27);
                auto tmp29 = std::abs(tmp28);
                auto tmp30 = static_cast<float>(1.0);
                auto tmp31 = min_propagate_nan(tmp29, tmp30);
                auto tmp32 = tmp30 - tmp31;
                auto tmp33 = static_cast<float>(0.0);
                auto tmp34 = tmp22 ? tmp32 : tmp33;
                auto tmp36 = tmp34 / tmp35;
                out_ptr4[static_cast<long>(i1 + (270L*i0))] = tmp36;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(6L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(270L); i1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i2=static_cast<long>(0L); i2<static_cast<long>(270L); i2+=static_cast<long>(1L))
                {
                    auto tmp61 = out_ptr4[static_cast<long>(i1)];
                    auto tmp63 = out_ptr4[static_cast<long>(270L + i1)];
                    auto tmp66 = out_ptr4[static_cast<long>(540L + i1)];
                    auto tmp69 = out_ptr4[static_cast<long>(810L + i1)];
                    auto tmp72 = out_ptr4[static_cast<long>(1080L + i1)];
                    auto tmp0 = static_cast<long>(i1);
                    auto tmp1 = static_cast<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(1.2777777777777777);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = static_cast<float>(1.2777777910232544);
                    auto tmp7 = tmp5 - tmp6;
                    auto tmp8 = tmp7 + tmp2;
                    auto tmp9 = static_cast<long>(tmp8);
                    auto tmp10 = static_cast<long>(0);
                    auto tmp11 = max_propagate_nan(tmp9, tmp10);
                    auto tmp12 = tmp11 + tmp10;
                    auto tmp13 = static_cast<long>(344);
                    auto tmp14 = min_propagate_nan(tmp12, tmp13);
                    auto tmp15 = in_out_ptr0[static_cast<long>(i2 + (270L*tmp14) + (93150L*i0))];
                    auto tmp16 = static_cast<long>(i2);
                    auto tmp17 = static_cast<float>(tmp16);
                    auto tmp18 = tmp17 + tmp2;
                    auto tmp19 = static_cast<float>(1.6888888888888889);
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = static_cast<float>(1.6888889074325562);
                    auto tmp22 = tmp20 - tmp21;
                    auto tmp23 = tmp22 + tmp2;
                    auto tmp24 = static_cast<long>(tmp23);
                    auto tmp25 = max_propagate_nan(tmp24, tmp10);
                    auto tmp26 = static_cast<long>(4);
                    auto tmp27 = tmp25 + tmp26;
                    auto tmp28 = static_cast<long>(455);
                    auto tmp29 = min_propagate_nan(tmp27, tmp28);
                    auto tmp30 = in_ptr0[static_cast<long>(tmp29 + (456L*tmp14) + (157320L*i0))];
                    auto tmp31 = static_cast<float>(0.0);
                    auto tmp32 = decltype(tmp30)(tmp30 * tmp31);
                    auto tmp33 = tmp15 + tmp32;
                    auto tmp34 = static_cast<long>(1);
                    auto tmp35 = tmp11 + tmp34;
                    auto tmp36 = min_propagate_nan(tmp35, tmp13);
                    auto tmp37 = in_out_ptr0[static_cast<long>(i2 + (270L*tmp36) + (93150L*i0))];
                    auto tmp38 = in_ptr0[static_cast<long>(tmp29 + (456L*tmp36) + (157320L*i0))];
                    auto tmp39 = decltype(tmp38)(tmp38 * tmp31);
                    auto tmp40 = tmp37 + tmp39;
                    auto tmp41 = static_cast<long>(2);
                    auto tmp42 = tmp11 + tmp41;
                    auto tmp43 = min_propagate_nan(tmp42, tmp13);
                    auto tmp44 = in_out_ptr0[static_cast<long>(i2 + (270L*tmp43) + (93150L*i0))];
                    auto tmp45 = in_ptr0[static_cast<long>(tmp29 + (456L*tmp43) + (157320L*i0))];
                    auto tmp46 = decltype(tmp45)(tmp45 * tmp31);
                    auto tmp47 = tmp44 + tmp46;
                    auto tmp48 = static_cast<long>(3);
                    auto tmp49 = tmp11 + tmp48;
                    auto tmp50 = min_propagate_nan(tmp49, tmp13);
                    auto tmp51 = in_out_ptr0[static_cast<long>(i2 + (270L*tmp50) + (93150L*i0))];
                    auto tmp52 = in_ptr0[static_cast<long>(tmp29 + (456L*tmp50) + (157320L*i0))];
                    auto tmp53 = decltype(tmp52)(tmp52 * tmp31);
                    auto tmp54 = tmp51 + tmp53;
                    auto tmp55 = tmp11 + tmp26;
                    auto tmp56 = min_propagate_nan(tmp55, tmp13);
                    auto tmp57 = in_out_ptr0[static_cast<long>(i2 + (270L*tmp56) + (93150L*i0))];
                    auto tmp58 = in_ptr0[static_cast<long>(tmp29 + (456L*tmp56) + (157320L*i0))];
                    auto tmp59 = decltype(tmp58)(tmp58 * tmp31);
                    auto tmp60 = tmp57 + tmp59;
                    auto tmp62 = decltype(tmp33)(tmp33 * tmp61);
                    auto tmp64 = decltype(tmp40)(tmp40 * tmp63);
                    auto tmp65 = tmp62 + tmp64;
                    auto tmp67 = decltype(tmp47)(tmp47 * tmp66);
                    auto tmp68 = tmp65 + tmp67;
                    auto tmp70 = decltype(tmp54)(tmp54 * tmp69);
                    auto tmp71 = tmp68 + tmp70;
                    auto tmp73 = decltype(tmp60)(tmp60 * tmp72);
                    auto tmp74 = tmp71 + tmp73;
                    in_out_ptr1[static_cast<long>(i2 + (270L*i1) + (72900L*i0))] = tmp74;
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
    assert_size_stride(arg0_1, (2, 3, 345, 456), (471960, 157320, 456, 1))
    buf1 = empty_strided((270, ), (1, ), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((270, ), (1, ), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((5, 270), (270, 1), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((2, 3, 345, 270), (279450, 93150, 270, 1), device='cpu', dtype=torch.float32)
    buf6 = buf5; del buf5  # reuse
    buf8 = empty_strided((5, 270), (270, 1), device='cpu', dtype=torch.float32)
    buf10 = empty_strided((2, 3, 270, 270), (218700, 72900, 270, 1), device='cpu', dtype=torch.float32)
    buf13 = buf10; del buf10  # reuse
    cpp_fused_add_div_index_mul_sum_unbind_0(c_void_p(buf6.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf8.data_ptr()))
    del arg0_1
    return (buf13, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 3, 345, 456), (471960, 157320, 456, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
