
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
                    auto tmp6 = c10::convert<int>(x0);
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
                    auto tmp27 = c10::convert<int>(x0);
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
                    auto tmp6 = c10::convert<int>(x0);
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
                    auto tmp27 = c10::convert<int>(x0);
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
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(256L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(2L*x1)];
                    auto tmp1 = out_ptr1[static_cast<long>(2L*x2)];
                    auto tmp29 = out_ptr1[static_cast<long>(1L + (2L*x2))];
                    auto tmp38 = out_ptr0[static_cast<long>(1L + (2L*x1))];
                    auto tmp2 = c10::convert<int>(x1);
                    auto tmp3 = c10::convert<float>(tmp2);
                    auto tmp4 = static_cast<float>(0.5);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = static_cast<float>(1.953125);
                    auto tmp7 = decltype(tmp5)(tmp5 * tmp6);
                    auto tmp8 = decltype(tmp7)(tmp7 - tmp4);
                    auto tmp9 = static_cast<float>(0.0);
                    auto tmp10 = max_propagate_nan(tmp8, tmp9);
                    auto tmp11 = c10::convert<int>(tmp10);
                    auto tmp12 = static_cast<int>(0);
                    auto tmp13 = decltype(tmp11)(tmp11 + tmp12);
                    auto tmp14 = static_cast<int>(499);
                    auto tmp15 = min_propagate_nan(tmp13, tmp14);
                    auto tmp16 = c10::convert<int>(x2);
                    auto tmp17 = c10::convert<float>(tmp16);
                    auto tmp18 = decltype(tmp17)(tmp17 + tmp4);
                    auto tmp19 = decltype(tmp18)(tmp18 * tmp6);
                    auto tmp20 = decltype(tmp19)(tmp19 - tmp4);
                    auto tmp21 = max_propagate_nan(tmp20, tmp9);
                    auto tmp22 = c10::convert<int>(tmp21);
                    auto tmp23 = decltype(tmp22)(tmp22 + tmp12);
                    auto tmp24 = min_propagate_nan(tmp23, tmp14);
                    auto tmp25 = in_ptr0[static_cast<long>(tmp24 + (500L*tmp15) + (250000L*x0))];
                    auto tmp26 = c10::convert<float>(tmp25);
                    auto tmp27 = decltype(tmp1)(tmp1 * tmp26);
                    auto tmp28 = decltype(tmp0)(tmp0 * tmp27);
                    auto tmp30 = static_cast<int>(1);
                    auto tmp31 = decltype(tmp22)(tmp22 + tmp30);
                    auto tmp32 = min_propagate_nan(tmp31, tmp14);
                    auto tmp33 = in_ptr0[static_cast<long>(tmp32 + (500L*tmp15) + (250000L*x0))];
                    auto tmp34 = c10::convert<float>(tmp33);
                    auto tmp35 = decltype(tmp29)(tmp29 * tmp34);
                    auto tmp36 = decltype(tmp0)(tmp0 * tmp35);
                    auto tmp37 = decltype(tmp28)(tmp28 + tmp36);
                    auto tmp39 = decltype(tmp11)(tmp11 + tmp30);
                    auto tmp40 = min_propagate_nan(tmp39, tmp14);
                    auto tmp41 = in_ptr0[static_cast<long>(tmp24 + (500L*tmp40) + (250000L*x0))];
                    auto tmp42 = c10::convert<float>(tmp41);
                    auto tmp43 = decltype(tmp1)(tmp1 * tmp42);
                    auto tmp44 = decltype(tmp38)(tmp38 * tmp43);
                    auto tmp45 = decltype(tmp37)(tmp37 + tmp44);
                    auto tmp46 = in_ptr0[static_cast<long>(tmp32 + (500L*tmp40) + (250000L*x0))];
                    auto tmp47 = c10::convert<float>(tmp46);
                    auto tmp48 = decltype(tmp29)(tmp29 * tmp47);
                    auto tmp49 = decltype(tmp38)(tmp38 * tmp48);
                    auto tmp50 = decltype(tmp45)(tmp45 + tmp49);
                    out_ptr2[static_cast<long>(x2 + (256L*x1) + (65536L*x0))] = tmp50;
                }
            }
        }
    }
    {
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(393216L); x0+=static_cast<long>(8L))
        {
            auto tmp0 = at::vec::Vectorized<float>::loadu(out_ptr2 + static_cast<long>(x0));
            auto tmp1 = tmp0.round();
            auto tmp2 = at::vec::convert_float_to_uint8(tmp1);
            tmp2.store(out_ptr3 + static_cast<long>(x0), 8);
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
    buf2 = empty((2, 3, 256, 256), device='cpu', dtype=torch.float32)
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
