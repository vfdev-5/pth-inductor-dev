
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


cpp_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/2l/c2ljzlm4sosod7u6lyrroqdba6hmfcyijrric6p4t3fhbcmw6osp.h"
extern "C" void kernel(float* in_out_ptr0,
                       const unsigned char* in_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3,
                       unsigned char* out_ptr4)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(6L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<unsigned char>(x1);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.953125);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = static_cast<float>(0.0);
                    auto tmp8 = max_propagate_nan(tmp6, tmp7);
                    auto tmp9 = c10::convert<long>(tmp8);
                    auto tmp10 = c10::convert<unsigned char>(x2);
                    auto tmp11 = c10::convert<float>(tmp10);
                    auto tmp12 = decltype(tmp11)(tmp11 + tmp2);
                    auto tmp13 = decltype(tmp12)(tmp12 * tmp4);
                    auto tmp14 = decltype(tmp13)(tmp13 - tmp2);
                    auto tmp15 = max_propagate_nan(tmp14, tmp7);
                    auto tmp16 = c10::convert<long>(tmp15);
                    auto tmp17 = in_ptr0[static_cast<long>(tmp16 + (500L*tmp9) + (250000L*x0))];
                    auto tmp18 = c10::convert<float>(tmp17);
                    auto tmp19 = c10::convert<float>(tmp9);
                    auto tmp20 = decltype(tmp8)(tmp8 - tmp19);
                    auto tmp21 = static_cast<float>(1.0);
                    auto tmp22 = decltype(tmp21)(tmp21 - tmp20);
                    auto tmp23 = decltype(tmp18)(tmp18 * tmp22);
                    out_ptr0[static_cast<long>(x2 + (256L*x1) + (65536L*x0))] = tmp23;
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
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<unsigned char>(x1);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.953125);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = static_cast<float>(0.0);
                    auto tmp8 = max_propagate_nan(tmp6, tmp7);
                    auto tmp9 = std::ceil(tmp8);
                    auto tmp10 = static_cast<float>(499.0);
                    auto tmp11 = min_propagate_nan(tmp9, tmp10);
                    auto tmp12 = c10::convert<long>(tmp11);
                    auto tmp13 = c10::convert<unsigned char>(x2);
                    auto tmp14 = c10::convert<float>(tmp13);
                    auto tmp15 = decltype(tmp14)(tmp14 + tmp2);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp4);
                    auto tmp17 = decltype(tmp16)(tmp16 - tmp2);
                    auto tmp18 = max_propagate_nan(tmp17, tmp7);
                    auto tmp19 = c10::convert<long>(tmp18);
                    auto tmp20 = in_ptr0[static_cast<long>(tmp19 + (500L*tmp12) + (250000L*x0))];
                    auto tmp21 = c10::convert<float>(tmp20);
                    auto tmp22 = c10::convert<long>(tmp8);
                    auto tmp23 = c10::convert<float>(tmp22);
                    auto tmp24 = decltype(tmp8)(tmp8 - tmp23);
                    auto tmp25 = decltype(tmp21)(tmp21 * tmp24);
                    out_ptr1[static_cast<long>(x2 + (256L*x1) + (65536L*x0))] = tmp25;
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
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<unsigned char>(x1);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.953125);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = static_cast<float>(0.0);
                    auto tmp8 = max_propagate_nan(tmp6, tmp7);
                    auto tmp9 = c10::convert<long>(tmp8);
                    auto tmp10 = c10::convert<unsigned char>(x2);
                    auto tmp11 = c10::convert<float>(tmp10);
                    auto tmp12 = decltype(tmp11)(tmp11 + tmp2);
                    auto tmp13 = decltype(tmp12)(tmp12 * tmp4);
                    auto tmp14 = decltype(tmp13)(tmp13 - tmp2);
                    auto tmp15 = max_propagate_nan(tmp14, tmp7);
                    auto tmp16 = std::ceil(tmp15);
                    auto tmp17 = static_cast<float>(499.0);
                    auto tmp18 = min_propagate_nan(tmp16, tmp17);
                    auto tmp19 = c10::convert<long>(tmp18);
                    auto tmp20 = in_ptr0[static_cast<long>(tmp19 + (500L*tmp9) + (250000L*x0))];
                    auto tmp21 = c10::convert<float>(tmp20);
                    auto tmp22 = c10::convert<float>(tmp9);
                    auto tmp23 = decltype(tmp8)(tmp8 - tmp22);
                    auto tmp24 = static_cast<float>(1.0);
                    auto tmp25 = decltype(tmp24)(tmp24 - tmp23);
                    auto tmp26 = decltype(tmp21)(tmp21 * tmp25);
                    out_ptr2[static_cast<long>(x2 + (256L*x1) + (65536L*x0))] = tmp26;
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
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<unsigned char>(x1);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = static_cast<float>(1.953125);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = static_cast<float>(0.0);
                    auto tmp8 = max_propagate_nan(tmp6, tmp7);
                    auto tmp9 = std::ceil(tmp8);
                    auto tmp10 = static_cast<float>(499.0);
                    auto tmp11 = min_propagate_nan(tmp9, tmp10);
                    auto tmp12 = c10::convert<long>(tmp11);
                    auto tmp13 = c10::convert<unsigned char>(x2);
                    auto tmp14 = c10::convert<float>(tmp13);
                    auto tmp15 = decltype(tmp14)(tmp14 + tmp2);
                    auto tmp16 = decltype(tmp15)(tmp15 * tmp4);
                    auto tmp17 = decltype(tmp16)(tmp16 - tmp2);
                    auto tmp18 = max_propagate_nan(tmp17, tmp7);
                    auto tmp19 = std::ceil(tmp18);
                    auto tmp20 = min_propagate_nan(tmp19, tmp10);
                    auto tmp21 = c10::convert<long>(tmp20);
                    auto tmp22 = in_ptr0[static_cast<long>(tmp21 + (500L*tmp12) + (250000L*x0))];
                    auto tmp23 = c10::convert<float>(tmp22);
                    auto tmp24 = c10::convert<long>(tmp8);
                    auto tmp25 = c10::convert<float>(tmp24);
                    auto tmp26 = decltype(tmp8)(tmp8 - tmp25);
                    auto tmp27 = decltype(tmp23)(tmp23 * tmp26);
                    out_ptr3[static_cast<long>(x2 + (256L*x1) + (65536L*x0))] = tmp27;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1536L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(256L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr0[static_cast<long>(x1 + (256L*x0))];
                auto tmp1 = out_ptr1[static_cast<long>(x1 + (256L*x0))];
                auto tmp18 = out_ptr2[static_cast<long>(x1 + (256L*x0))];
                auto tmp19 = out_ptr3[static_cast<long>(x1 + (256L*x0))];
                auto tmp2 = decltype(tmp0)(tmp0 + tmp1);
                auto tmp3 = c10::convert<unsigned char>(x1);
                auto tmp4 = c10::convert<float>(tmp3);
                auto tmp5 = static_cast<float>(0.5);
                auto tmp6 = decltype(tmp4)(tmp4 + tmp5);
                auto tmp7 = static_cast<float>(1.953125);
                auto tmp8 = decltype(tmp6)(tmp6 * tmp7);
                auto tmp9 = decltype(tmp8)(tmp8 - tmp5);
                auto tmp10 = static_cast<float>(0.0);
                auto tmp11 = max_propagate_nan(tmp9, tmp10);
                auto tmp12 = c10::convert<long>(tmp11);
                auto tmp13 = c10::convert<float>(tmp12);
                auto tmp14 = decltype(tmp11)(tmp11 - tmp13);
                auto tmp15 = static_cast<float>(1.0);
                auto tmp16 = decltype(tmp15)(tmp15 - tmp14);
                auto tmp17 = decltype(tmp2)(tmp2 * tmp16);
                auto tmp20 = decltype(tmp18)(tmp18 + tmp19);
                auto tmp21 = decltype(tmp20)(tmp20 * tmp14);
                auto tmp22 = decltype(tmp17)(tmp17 + tmp21);
                in_out_ptr0[static_cast<long>(x1 + (256L*x0))] = tmp22;
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(in_out_ptr0 + static_cast<long>(x0));
            auto tmp1 = at::vec::convert_float_to_uint8(tmp0);
            tmp1.store(out_ptr4 + static_cast<long>(x0), 8);
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
    buf0 = empty((2, 3, 256, 256), device='cpu', dtype=torch.float32)
    buf1 = empty((2, 3, 256, 256), device='cpu', dtype=torch.float32)
    buf2 = empty((2, 3, 256, 256), device='cpu', dtype=torch.float32)
    buf3 = empty((2, 3, 256, 256), device='cpu', dtype=torch.float32)
    buf4 = buf0; del buf0  # reuse
    buf5 = empty((2, 3, 256, 256), device='cpu', dtype=torch.uint8)
    cpp_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_0(c_void_p(buf4.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf5.data_ptr()))
    del arg0_1
    return (buf5, )


def benchmark_compiled_module(times=10000, repeat=500):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((2, 3, 500, 500), (750000, 250000, 500, 1), device='cpu', dtype=torch.uint8)
    return print_performance(lambda: call([arg0_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
