
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
                       const long ks4)
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
                    auto tmp31 = c10::convert<float>(ks1*(1.0/ks0));
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
                    auto tmp10 = c10::convert<float>(ks1*(1.0/ks2));
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
                    auto tmp31 = c10::convert<float>(ks1*(1.0/ks2));
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks3*ks4); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(ks0); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(ks2); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(2L*x1)];
                    auto tmp1 = out_ptr1[static_cast<long>(2L*x2)];
                    auto tmp32 = out_ptr1[static_cast<long>(1L + (2L*x2))];
                    auto tmp41 = out_ptr0[static_cast<long>(1L + (2L*x1))];
                    auto tmp2 = c10::convert<long>(x1);
                    auto tmp3 = c10::convert<float>(tmp2);
                    auto tmp4 = static_cast<float>(0.5);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(ks1*(1.0/ks0));
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp8 = decltype(tmp7)(tmp7 - tmp4);
                    auto tmp9 = static_cast<float>(0.0);
                    auto tmp10 = max_propagate_nan(tmp8, tmp9);
                    auto tmp11 = c10::convert<int>(tmp10);
                    auto tmp12 = c10::convert<long>(tmp11);
                    auto tmp13 = static_cast<long>(0);
                    auto tmp14 = decltype(tmp12)(tmp12 + tmp13);
                    auto tmp15 = c10::convert<long>((-1L) + ks1);
                    auto tmp16 = min_propagate_nan(tmp14, tmp15);
                    auto tmp17 = c10::convert<long>(x2);
                    auto tmp18 = c10::convert<float>(tmp17);
                    auto tmp19 = decltype(tmp18)(tmp18 + tmp4);
                    auto tmp20 = c10::convert<float>(ks1*(1.0/ks2));
                    auto tmp21 = decltype(tmp19)(tmp19 * tmp20);
                    auto tmp22 = decltype(tmp21)(tmp21 - tmp4);
                    auto tmp23 = max_propagate_nan(tmp22, tmp9);
                    auto tmp24 = c10::convert<int>(tmp23);
                    auto tmp25 = c10::convert<long>(tmp24);
                    auto tmp26 = decltype(tmp25)(tmp25 + tmp13);
                    auto tmp27 = min_propagate_nan(tmp26, tmp15);
                    auto tmp28 = in_ptr0[static_cast<long>(tmp27 + (ks1*tmp16) + (x0*(static_cast<long>(ks1*ks1))))];
                    auto tmp29 = c10::convert<float>(tmp28);
                    auto tmp30 = decltype(tmp1)(tmp1 * tmp29);
                    auto tmp31 = decltype(tmp0)(tmp0 * tmp30);
                    auto tmp33 = static_cast<long>(1);
                    auto tmp34 = decltype(tmp25)(tmp25 + tmp33);
                    auto tmp35 = min_propagate_nan(tmp34, tmp15);
                    auto tmp36 = in_ptr0[static_cast<long>(tmp35 + (ks1*tmp16) + (x0*(static_cast<long>(ks1*ks1))))];
                    auto tmp37 = c10::convert<float>(tmp36);
                    auto tmp38 = decltype(tmp32)(tmp32 * tmp37);
                    auto tmp39 = decltype(tmp0)(tmp0 * tmp38);
                    auto tmp40 = decltype(tmp31)(tmp31 + tmp39);
                    auto tmp42 = decltype(tmp12)(tmp12 + tmp33);
                    auto tmp43 = min_propagate_nan(tmp42, tmp15);
                    auto tmp44 = in_ptr0[static_cast<long>(tmp27 + (ks1*tmp43) + (x0*(static_cast<long>(ks1*ks1))))];
                    auto tmp45 = c10::convert<float>(tmp44);
                    auto tmp46 = decltype(tmp1)(tmp1 * tmp45);
                    auto tmp47 = decltype(tmp41)(tmp41 * tmp46);
                    auto tmp48 = decltype(tmp40)(tmp40 + tmp47);
                    auto tmp49 = in_ptr0[static_cast<long>(tmp35 + (ks1*tmp43) + (x0*(static_cast<long>(ks1*ks1))))];
                    auto tmp50 = c10::convert<float>(tmp49);
                    auto tmp51 = decltype(tmp32)(tmp32 * tmp50);
                    auto tmp52 = decltype(tmp41)(tmp41 * tmp51);
                    auto tmp53 = decltype(tmp48)(tmp48 + tmp52);
                    out_ptr2[static_cast<long>(x2 + (ks2*x1) + (ks0*ks2*x0))] = tmp53;
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(8L*(c10::div_floor_integer((ks0*ks2*ks3*ks4), 8L))); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
            auto tmp1 = tmp0.round();
            auto tmp2 = at::vec::convert_float_to_uint8(tmp1);
            tmp2.store(out_ptr3 + static_cast<long>(x0), 8);
        }
        #pragma omp simd simdlen(4)
        for(long x0=static_cast<long>(8L*(c10::div_floor_integer((ks0*ks2*ks3*ks4), 8L))); x0<static_cast<long>(ks0*ks2*ks3*ks4); x0+=static_cast<long>(1L))
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    s3 = arg4_1
    s4 = arg5_1
    assert_size_stride(arg3_1, (s0, s1, s2, s2), (s1*(s2*s2), s2*s2, s2, 1))
    buf0 = empty_strided((s3, 2), (2, 1), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((s4, 2), (2, 1), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((s0, s1, s3, s4), (s3*s4*s1, s3*s4, s4, 1), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((s0, s1, s3, s4), (s3*s4*s1, s3*s4, s4, 1), device='cpu', dtype=torch.uint8)
    cpp_fused__to_copy__unsafe_index_mul_round_stack_sum_0(c_void_p(arg3_1.data_ptr()), c_void_p(buf0.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_long(s3), c_long(s2), c_long(s4), c_long(s0), c_long(s1))
    del arg3_1
    return (buf3, )


def benchmark_compiled_module(times=10000, repeat=1000):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 2
    arg1_1 = 3
    arg2_1 = 500
    arg3_1 = rand_strided((2, 3, 500, 500), (750000, 250000, 500, 1), device='cpu', dtype=torch.uint8)
    arg4_1 = 256
    arg5_1 = 256
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
