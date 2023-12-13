
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


cpp_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       const long ks0,
                       const long ks1,
                       const long ks2,
                       const long ks3,
                       const long ks4)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(ks1); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(ks2); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = c10::convert<float>(((-1L)*(1.0/((-1L) + ks1))) + (ks3*(1.0/((-1L) + ks1))));
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = c10::convert<long>(tmp8);
                    auto tmp10 = decltype(tmp9)(tmp9 + ks3);
                    auto tmp11 = tmp9 < 0;
                    auto tmp12 = tmp11 ? tmp10 : tmp9;
                    auto tmp13 = c10::convert<long>(x2);
                    auto tmp14 = c10::convert<double>(tmp13);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp2);
                    auto tmp16 = decltype(tmp15)(tmp15 + tmp4);
                    auto tmp17 = c10::convert<float>(tmp16);
                    auto tmp18 = c10::convert<float>(((-1L)*(1.0/((-1L) + ks2))) + (ks4*(1.0/((-1L) + ks2))));
                    auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                    auto tmp20 = c10::convert<long>(tmp19);
                    auto tmp21 = decltype(tmp20)(tmp20 + ks4);
                    auto tmp22 = tmp20 < 0;
                    auto tmp23 = tmp22 ? tmp21 : tmp20;
                    auto tmp24 = in_ptr0[static_cast<long>(tmp23 + (ks4*tmp12) + (ks3*ks4*x0))];
                    auto tmp25 = c10::convert<float>(tmp9);
                    auto tmp26 = decltype(tmp8)(tmp8 - tmp25);
                    auto tmp27 = static_cast<float>(1.0);
                    auto tmp28 = decltype(tmp27)(tmp27 - tmp26);
                    auto tmp29 = decltype(tmp24)(tmp24 * tmp28);
                    out_ptr0[static_cast<long>(x2 + (ks2*x1) + (ks1*ks2*x0))] = tmp29;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(ks1); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(ks2); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = c10::convert<float>(((-1L)*(1.0/((-1L) + ks1))) + (ks3*(1.0/((-1L) + ks1))));
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = std::ceil(tmp8);
                    auto tmp10 = c10::convert<float>((-1L) + ks3);
                    auto tmp11 = min_propagate_nan(tmp9, tmp10);
                    auto tmp12 = c10::convert<long>(tmp11);
                    auto tmp13 = decltype(tmp12)(tmp12 + ks3);
                    auto tmp14 = tmp12 < 0;
                    auto tmp15 = tmp14 ? tmp13 : tmp12;
                    auto tmp16 = c10::convert<long>(x2);
                    auto tmp17 = c10::convert<double>(tmp16);
                    auto tmp18 = decltype(tmp17)(tmp17 * tmp2);
                    auto tmp19 = decltype(tmp18)(tmp18 + tmp4);
                    auto tmp20 = c10::convert<float>(tmp19);
                    auto tmp21 = c10::convert<float>(((-1L)*(1.0/((-1L) + ks2))) + (ks4*(1.0/((-1L) + ks2))));
                    auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                    auto tmp23 = c10::convert<long>(tmp22);
                    auto tmp24 = decltype(tmp23)(tmp23 + ks4);
                    auto tmp25 = tmp23 < 0;
                    auto tmp26 = tmp25 ? tmp24 : tmp23;
                    auto tmp27 = in_ptr0[static_cast<long>(tmp26 + (ks4*tmp15) + (ks3*ks4*x0))];
                    auto tmp28 = c10::convert<long>(tmp8);
                    auto tmp29 = c10::convert<float>(tmp28);
                    auto tmp30 = decltype(tmp8)(tmp8 - tmp29);
                    auto tmp31 = decltype(tmp27)(tmp27 * tmp30);
                    out_ptr1[static_cast<long>(x2 + (ks2*x1) + (ks1*ks2*x0))] = tmp31;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(ks1); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(ks2); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = c10::convert<float>(((-1L)*(1.0/((-1L) + ks1))) + (ks3*(1.0/((-1L) + ks1))));
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = c10::convert<long>(tmp8);
                    auto tmp10 = decltype(tmp9)(tmp9 + ks3);
                    auto tmp11 = tmp9 < 0;
                    auto tmp12 = tmp11 ? tmp10 : tmp9;
                    auto tmp13 = c10::convert<long>(x2);
                    auto tmp14 = c10::convert<double>(tmp13);
                    auto tmp15 = decltype(tmp14)(tmp14 * tmp2);
                    auto tmp16 = decltype(tmp15)(tmp15 + tmp4);
                    auto tmp17 = c10::convert<float>(tmp16);
                    auto tmp18 = c10::convert<float>(((-1L)*(1.0/((-1L) + ks2))) + (ks4*(1.0/((-1L) + ks2))));
                    auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                    auto tmp20 = std::ceil(tmp19);
                    auto tmp21 = c10::convert<float>((-1L) + ks4);
                    auto tmp22 = min_propagate_nan(tmp20, tmp21);
                    auto tmp23 = c10::convert<long>(tmp22);
                    auto tmp24 = decltype(tmp23)(tmp23 + ks4);
                    auto tmp25 = tmp23 < 0;
                    auto tmp26 = tmp25 ? tmp24 : tmp23;
                    auto tmp27 = in_ptr0[static_cast<long>(tmp26 + (ks4*tmp12) + (ks3*ks4*x0))];
                    auto tmp28 = c10::convert<float>(tmp9);
                    auto tmp29 = decltype(tmp8)(tmp8 - tmp28);
                    auto tmp30 = static_cast<float>(1.0);
                    auto tmp31 = decltype(tmp30)(tmp30 - tmp29);
                    auto tmp32 = decltype(tmp27)(tmp27 * tmp31);
                    out_ptr2[static_cast<long>(x2 + (ks2*x1) + (ks1*ks2*x0))] = tmp32;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(ks1); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(ks2); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = c10::convert<float>(((-1L)*(1.0/((-1L) + ks1))) + (ks3*(1.0/((-1L) + ks1))));
                    auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                    auto tmp9 = std::ceil(tmp8);
                    auto tmp10 = c10::convert<float>((-1L) + ks3);
                    auto tmp11 = min_propagate_nan(tmp9, tmp10);
                    auto tmp12 = c10::convert<long>(tmp11);
                    auto tmp13 = decltype(tmp12)(tmp12 + ks3);
                    auto tmp14 = tmp12 < 0;
                    auto tmp15 = tmp14 ? tmp13 : tmp12;
                    auto tmp16 = c10::convert<long>(x2);
                    auto tmp17 = c10::convert<double>(tmp16);
                    auto tmp18 = decltype(tmp17)(tmp17 * tmp2);
                    auto tmp19 = decltype(tmp18)(tmp18 + tmp4);
                    auto tmp20 = c10::convert<float>(tmp19);
                    auto tmp21 = c10::convert<float>(((-1L)*(1.0/((-1L) + ks2))) + (ks4*(1.0/((-1L) + ks2))));
                    auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                    auto tmp23 = std::ceil(tmp22);
                    auto tmp24 = c10::convert<float>((-1L) + ks4);
                    auto tmp25 = min_propagate_nan(tmp23, tmp24);
                    auto tmp26 = c10::convert<long>(tmp25);
                    auto tmp27 = decltype(tmp26)(tmp26 + ks4);
                    auto tmp28 = tmp26 < 0;
                    auto tmp29 = tmp28 ? tmp27 : tmp26;
                    auto tmp30 = in_ptr0[static_cast<long>(tmp29 + (ks4*tmp15) + (ks3*ks4*x0))];
                    auto tmp31 = c10::convert<long>(tmp8);
                    auto tmp32 = c10::convert<float>(tmp31);
                    auto tmp33 = decltype(tmp8)(tmp8 - tmp32);
                    auto tmp34 = decltype(tmp30)(tmp30 * tmp33);
                    out_ptr3[static_cast<long>(x2 + (ks2*x1) + (ks1*ks2*x0))] = tmp34;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0*ks1); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(ks2); x1+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr0[static_cast<long>(x1 + (ks2*x0))];
                auto tmp1 = out_ptr1[static_cast<long>(x1 + (ks2*x0))];
                auto tmp18 = out_ptr2[static_cast<long>(x1 + (ks2*x0))];
                auto tmp19 = out_ptr3[static_cast<long>(x1 + (ks2*x0))];
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                auto tmp3 = c10::convert<long>(x1);
                auto tmp4 = c10::convert<double>(tmp3);
                auto tmp5 = static_cast<double>(1.0);
                auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                auto tmp7 = static_cast<double>(0.0);
                auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                auto tmp9 = c10::convert<float>(tmp8);
                auto tmp10 = c10::convert<float>(((-1L)*(1.0/((-1L) + ks2))) + (ks4*(1.0/((-1L) + ks2))));
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp12 = c10::convert<long>(tmp11);
                auto tmp13 = c10::convert<float>(tmp12);
                auto tmp14 = decltype(tmp11)(tmp11 - tmp13);
                auto tmp15 = static_cast<float>(1.0);
                auto tmp16 = decltype(tmp15)(tmp15 - tmp14);
                auto tmp17 = decltype(tmp2)(tmp2 * tmp16);
                auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                auto tmp21 = decltype(tmp20)(tmp20 * tmp14);
                auto tmp22 = decltype(tmp17)(tmp17 + tmp21);
                in_out_ptr0[static_cast<long>(x1 + (ks2*x0))] = tmp22;
            }
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
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    buf0 = empty_strided((1, s0, s3, s4), (s3*s4*s0, s3*s4, s4, 1), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, s0, s3, s4), (s3*s4*s0, s3*s4, s4, 1), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((1, s0, s3, s4), (s3*s4*s0, s3*s4, s4, 1), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((1, s0, s3, s4), (s3*s4*s0, s3*s4, s4, 1), device='cpu', dtype=torch.float32)
    buf4 = buf0; del buf0  # reuse
    cpp_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_0(c_void_p(buf4.data_ptr()), c_void_p(arg3_1.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_long(s0), c_long(s3), c_long(s4), c_long(s1), c_long(s2))
    del arg3_1
    return (buf4, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 500
    arg2_1 = 400
    arg3_1 = rand_strided((1, 3, 500, 400), (600000, 200000, 400, 1), device='cpu', dtype=torch.float32)
    arg4_1 = 300
    arg5_1 = 256
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    # from torch._inductor.wrapper_benchmark import compiled_module_main
    # compiled_module_main('None', benchmark_compiled_module)
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 500
    arg2_1 = 400
    arg3_1 = rand_strided((1, 3, 500, 400), (600000, 200000, 400, 1), device='cpu', dtype=torch.float32)
    arg4_1 = 300
    arg5_1 = 256
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1])

    output = fn()
    expected = torch.nn.functional.interpolate(arg3_1, (arg4_1, arg5_1), mode="bilinear", antialias=False, align_corners=True)

    torch.testing.assert_close(output[0], expected)
