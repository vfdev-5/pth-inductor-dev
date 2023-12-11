
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
                       float* out_ptr2,
                       unsigned char* out_ptr3,
                       const long ks0,
                       const long ks1,
                       const long ks2,
                       const long ks3,
                       const long ks4,
                       const long ks5)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
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
                    auto tmp10 = c10::convert<float>(ks1*(1.0/ks0));
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp12 = decltype(tmp11)(tmp11 - tmp8);
                    auto tmp13 = static_cast<float>(0.0);
                    auto tmp14 = max_propagate_nan(tmp12, tmp13);
                    auto tmp15 = c10::convert<long>(tmp14);
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
                    auto tmp31 = c10::convert<float>(ks1*(1.0/ks0));
                    auto tmp32 = decltype(tmp30)(tmp30 * tmp31);
                    auto tmp33 = decltype(tmp32)(tmp32 - tmp29);
                    auto tmp34 = static_cast<float>(0.0);
                    auto tmp35 = max_propagate_nan(tmp33, tmp34);
                    auto tmp36 = c10::convert<long>(tmp35);
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks2); x0+=static_cast<long>(1L))
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
                    auto tmp10 = c10::convert<float>(ks3*(1.0/ks2));
                    auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                    auto tmp12 = decltype(tmp11)(tmp11 - tmp8);
                    auto tmp13 = static_cast<float>(0.0);
                    auto tmp14 = max_propagate_nan(tmp12, tmp13);
                    auto tmp15 = c10::convert<long>(tmp14);
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
                    auto tmp31 = c10::convert<float>(ks3*(1.0/ks2));
                    auto tmp32 = decltype(tmp30)(tmp30 * tmp31);
                    auto tmp33 = decltype(tmp32)(tmp32 - tmp29);
                    auto tmp34 = static_cast<float>(0.0);
                    auto tmp35 = max_propagate_nan(tmp33, tmp34);
                    auto tmp36 = c10::convert<long>(tmp35);
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks4*ks5); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(ks0); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(ks2); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(2L*x1)];
                    auto tmp1 = out_ptr1[static_cast<long>(2L*x2)];
                    auto tmp31 = out_ptr1[static_cast<long>(1L + (2L*x2))];
                    auto tmp40 = out_ptr0[static_cast<long>(1L + (2L*x1))];
                    auto tmp2 = c10::convert<long>(x1);
                    auto tmp3 = c10::convert<float>(tmp2);
                    auto tmp4 = static_cast<float>(0.5);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(ks1*(1.0/ks0));
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp8 = decltype(tmp7)(tmp7 - tmp4);
                    auto tmp9 = static_cast<float>(0.0);
                    auto tmp10 = max_propagate_nan(tmp8, tmp9);
                    auto tmp11 = c10::convert<long>(tmp10);
                    auto tmp12 = static_cast<long>(0);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = c10::convert<long>((-1L) + ks1);
                    auto tmp15 = min_propagate_nan(tmp13, tmp14);
                    auto tmp16 = c10::convert<long>(x2);
                    auto tmp17 = c10::convert<float>(tmp16);
                    auto tmp18 = decltype(tmp17)(tmp17 + tmp4);
                    auto tmp19 = c10::convert<float>(ks3*(1.0/ks2));
                    auto tmp20 = decltype(tmp18)(tmp18 * tmp19);
                    auto tmp21 = decltype(tmp20)(tmp20 - tmp4);
                    auto tmp22 = max_propagate_nan(tmp21, tmp9);
                    auto tmp23 = c10::convert<long>(tmp22);
                    auto tmp24 = decltype(tmp23)(tmp23 + tmp12);
                    auto tmp25 = c10::convert<long>((-1L) + ks3);
                    auto tmp26 = min_propagate_nan(tmp24, tmp25);
                    auto tmp27 = in_ptr0[static_cast<long>(tmp26 + (ks3*tmp15) + (ks1*ks3*x0))];
                    auto tmp28 = c10::convert<float>(tmp27);
                    auto tmp29 = decltype(tmp1)(tmp1 * tmp28);
                    auto tmp30 = decltype(tmp0)(tmp0 * tmp29);
                    auto tmp32 = static_cast<long>(1);
                    auto tmp33 = decltype(tmp23)(tmp23 + tmp32);
                    auto tmp34 = min_propagate_nan(tmp33, tmp25);
                    auto tmp35 = in_ptr0[static_cast<long>(tmp34 + (ks3*tmp15) + (ks1*ks3*x0))];
                    auto tmp36 = c10::convert<float>(tmp35);
                    auto tmp37 = decltype(tmp31)(tmp31 * tmp36);
                    auto tmp38 = decltype(tmp0)(tmp0 * tmp37);
                    auto tmp39 = decltype(tmp30)(tmp30 + tmp38);
                    auto tmp41 = decltype(tmp11)(tmp11 + tmp32);
                    auto tmp42 = min_propagate_nan(tmp41, tmp14);
                    auto tmp43 = in_ptr0[static_cast<long>(tmp26 + (ks3*tmp42) + (ks1*ks3*x0))];
                    auto tmp44 = c10::convert<float>(tmp43);
                    auto tmp45 = decltype(tmp1)(tmp1 * tmp44);
                    auto tmp46 = decltype(tmp40)(tmp40 * tmp45);
                    auto tmp47 = decltype(tmp39)(tmp39 + tmp46);
                    auto tmp48 = in_ptr0[static_cast<long>(tmp34 + (ks3*tmp42) + (ks1*ks3*x0))];
                    auto tmp49 = c10::convert<float>(tmp48);
                    auto tmp50 = decltype(tmp31)(tmp31 * tmp49);
                    auto tmp51 = decltype(tmp40)(tmp40 * tmp50);
                    auto tmp52 = decltype(tmp47)(tmp47 + tmp51);
                    out_ptr2[static_cast<long>(x2 + (ks2*x1) + (ks0*ks2*x0))] = tmp52;
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L*(c10::div_floor_integer((ks0*ks2*ks4*ks5), 8L))); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
            auto tmp1 = tmp0.round();
            auto tmp2 = at::vec::convert_float_to_uint8(tmp1);
            tmp2.store(out_ptr3 + static_cast<long>(x0), 8);
        }
        #pragma omp simd simdlen(4)
        for(long x0=static_cast<long>(8L*(c10::div_floor_integer((ks0*ks2*ks4*ks5), 8L))); x0<static_cast<long>(ks0*ks2*ks4*ks5); x0+=static_cast<long>(1L))
        {
            auto tmp0 = out_ptr2[static_cast<long>(x0)];
            auto tmp1 = std::nearbyint(tmp0);
            auto tmp2 = c10::convert<unsigned char>(tmp1);
            out_ptr3[static_cast<long>(x0)] = tmp2;
        }
    }
}
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    s3 = arg3_1
    s4 = arg5_1
    s5 = arg6_1
    assert_size_stride(arg4_1, (s0, s1, s2, s3), (s1*s2*s3, s2*s3, s3, 1))
    buf0 = empty_strided((s4, 2), (2, 1), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((s5, 2), (2, 1), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((s0, s1, s4, s5), (s4*s5*s1, s4*s5, s5, 1), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((s0, s1, s4, s5), (s4*s5*s1, s4*s5, s5, 1), device='cpu', dtype=torch.uint8)
    cpp_fused__to_copy__unsafe_index_mul_round_stack_sum_0(c_void_p(arg4_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_long(s4), c_long(s2), c_long(s5), c_long(s3), c_long(s0), c_long(s1))
    del arg4_1
    return (buf3, )


def benchmark_compiled_module(times=10000, repeat=500):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 2
    arg1_1 = 3
    arg2_1 = 500
    arg3_1 = 400
    arg4_1 = rand_strided((2, 3, 500, 400), (600000, 200000, 400, 1), device='cpu', dtype=torch.uint8)
    arg5_1 = 256
    arg6_1 = 256
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
