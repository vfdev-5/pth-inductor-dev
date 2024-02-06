
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


# cpp_fused_upsample_nearest2d_0 = async_compile.cpp('''
# #include "/tmp/torchinductor_root/gb/cgbau5vlj6cetmcjbjbtw6x4rrivaln6f45s5d72gy2bfx5foz3k.h"
# extern "C" void kernel(const float* in_ptr0,
#                        float* out_ptr0)
# {
#     {
#         #pragma GCC ivdep
#         for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(1L))
#         {
#             #pragma GCC ivdep
#             for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
#             {
#                 #pragma GCC ivdep
#                 for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
#                 {
#                     auto tmp0 = c10::convert<float>(x1);
#                     auto tmp1 = static_cast<float>(2.0);
#                     auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
#                     auto tmp3 = c10::convert<int>(tmp2);
#                     auto tmp4 = c10::convert<float>(x2);
#                     auto tmp5 = decltype(tmp4)(tmp4 * tmp1);
#                     auto tmp6 = c10::convert<int>(tmp5);
#                     auto tmp7 = in_ptr0[static_cast<long>(tmp6 + (4L*tmp3) + (16L*x0))];
#                     out_ptr0[static_cast<long>(x2 + (8L*x1) + (64L*x0))] = tmp7;
#                 }
#             }
#         }
#     }
# }
# ''')


cpp_fused_upsample_nearest2d_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/gb/cgbau5vlj6cetmcjbjbtw6x4rrivaln6f45s5d72gy2bfx5foz3k.h"
extern "C" void kernel(const float* in_ptr0, float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<float>(x1);
                    auto tmp1 = static_cast<float>(2.0);
                    auto tmp2 = decltype(tmp0)(tmp0 * tmp1);
                    auto tmp3 = c10::convert<int>(tmp2);
                    auto tmp4 = c10::convert<float>(x2);
                    auto tmp5 = decltype(tmp4)(tmp4 * tmp1);
                    auto tmp6 = c10::convert<int>(tmp5);
                    auto tmp7 = in_ptr0[static_cast<long>(tmp6 + (4L*tmp3) + (16L*x0))];
                    out_ptr0[static_cast<long>(x2 + (8L*x1) + (64L*x0))] = tmp7;
                }
            }
        }
    }
}
''')



cpp_fused__unsafe_index_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/gb/cgbau5vlj6cetmcjbjbtw6x4rrivaln6f45s5d72gy2bfx5foz3k.h"
extern "C" void kernel(const float* in_ptr0, float* out_ptr0)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1920L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(8L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(8L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = c10::convert<double>(tmp0);
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = decltype(tmp3)(tmp3 + tmp4);
                    auto tmp6 = c10::convert<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.0);
                    auto tmp8 = decltype(tmp6)(tmp6 + tmp7);
                    auto tmp9 = static_cast<float>(0.5);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = c10::convert<long>(tmp10);
                    auto tmp12 = c10::convert<long>(x2);
                    auto tmp13 = c10::convert<double>(tmp12);
                    auto tmp14 = decltype(tmp13)(tmp13 * tmp2);
                    auto tmp15 = decltype(tmp14)(tmp14 + tmp4);
                    auto tmp16 = c10::convert<float>(tmp15);
                    auto tmp17 = decltype(tmp16)(tmp16 + tmp7);
                    auto tmp18 = decltype(tmp17)(tmp17 * tmp9);
                    auto tmp19 = c10::convert<long>(tmp18);
                    auto tmp20 = in_ptr0[static_cast<long>(tmp19 + (4L*tmp11) + (16L*x0))];
                    out_ptr0[static_cast<long>(x2 + (8L*x1) + (64L*x0))] = tmp20;
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
    assert_size_stride(arg0_1, (3, 640, 4, 4), (10240, 16, 4, 1))
    buf0 = empty((3, 640, 8, 8), device='cpu', dtype=torch.float32)
    cpp_fused_upsample_nearest2d_0(c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()))
    del arg0_1
    return (buf0, )


def call_expected(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (3, 640, 4, 4), (10240, 16, 4, 1))
    buf0 = empty((3, 640, 8, 8), device='cpu', dtype=torch.float32)
    cpp_fused__unsafe_index_0(c_void_p(arg0_1.data_ptr()), c_void_p(buf0.data_ptr()))
    del arg0_1
    return (buf0, )


if __name__ == "__main__":
    from torch._dynamo.testing import rand_strided

    arg0_1 = rand_strided((3, 640, 4, 4), (10240, 16, 4, 1), device='cpu', dtype=torch.float32)

    output = call([arg0_1])
    expected = call_expected([arg0_1])

    torch.testing.assert_close(output, expected)

