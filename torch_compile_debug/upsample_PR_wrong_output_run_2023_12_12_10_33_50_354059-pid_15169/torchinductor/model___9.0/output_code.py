
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


cpp_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/26/c26eqbkuxvn72gf7p2xujmqjcwf4bo6lxmp6rwborxnf4gldnimh.h"
extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(ks1); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(ks2); x2+=static_cast<long>(1L))
                {
                    // auto tmp0 = c10::convert<long>(((-1.00000000000000)*(1.0/((-1.00000000000000) + ks1))) + (ks3*(1.0/((-1.00000000000000) + ks1))));
                    // auto tmp0 = ((-1.00000000000000)*(1.0/((-1.00000000000000) + ks1))) + (ks3*(1.0/((-1.00000000000000) + ks1)));
                    auto tmp0 = (ks3 - 1.0) / (ks1 - 1.0);

                    auto tmp1 = c10::convert<long>(x1);

                    // auto tmp2 = decltype(tmp1)(tmp1 * tmp0);
                    auto tmp2 = decltype(tmp0)(tmp1 * tmp0);

                    auto tmp3 = c10::convert<float>(tmp2);
                    auto tmp4 = static_cast<float>(0.0);
                    auto tmp5 = max_propagate_nan(tmp3, tmp4);
                    auto tmp6 = c10::convert<long>(tmp5);
                    auto tmp7 = c10::convert<long>((-1L) + ks3);
                    auto tmp8 = tmp6 < tmp7;
                    auto tmp9 = static_cast<long>(1);
                    auto tmp10 = decltype(tmp6)(tmp6 + tmp9);
                    auto tmp11 = tmp8 ? tmp10 : tmp6;

                    // auto tmp12 = c10::convert<long>(((-1.00000000000000)*(1.0/((-1.00000000000000) + ks2))) + (ks4*(1.0/((-1.00000000000000) + ks2))));
                    // auto tmp12 = ((-1.00000000000000)*(1.0/((-1.00000000000000) + ks2))) + (ks4*(1.0/((-1.00000000000000) + ks2)));
                    auto tmp12 = (ks4 - 1.0) / (ks2 - 1.0);

                    auto tmp13 = c10::convert<long>(x2);

                    // auto tmp14 = decltype(tmp13)(tmp13 * tmp12);
                    auto tmp14 = decltype(tmp12)(tmp13 * tmp12);

                    auto tmp15 = c10::convert<float>(tmp14);
                    auto tmp16 = max_propagate_nan(tmp15, tmp4);
                    auto tmp17 = c10::convert<long>(tmp16);
                    auto tmp18 = c10::convert<long>((-1L) + ks4);
                    auto tmp19 = tmp17 < tmp18;
                    auto tmp20 = decltype(tmp17)(tmp17 + tmp9);
                    auto tmp21 = tmp19 ? tmp20 : tmp17;

                    auto tmp22 = in_ptr0[static_cast<long>(tmp21 + (ks4*tmp11) + (ks3*ks4*x0))];
                    auto tmp23 = in_ptr0[static_cast<long>(tmp17 + (ks4*tmp11) + (ks3*ks4*x0))];

                    auto tmp24 = decltype(tmp22)(tmp22 - tmp23);
                    auto tmp25 = c10::convert<float>(tmp17);
                    auto tmp26 = decltype(tmp16)(tmp16 - tmp25);
                    auto tmp27 = max_propagate_nan(tmp26, tmp4);
                    auto tmp28 = static_cast<float>(1.0);
                    auto tmp29 = min_propagate_nan(tmp27, tmp28);
                    auto tmp30 = decltype(tmp24)(tmp24 * tmp29);

                    auto tmp31 = in_ptr0[static_cast<long>(tmp21 + (ks4*tmp6) + (ks3*ks4*x0))];
                    auto tmp32 = in_ptr0[static_cast<long>(tmp17 + (ks4*tmp6) + (ks3*ks4*x0))];
                    auto tmp33 = decltype(tmp31)(tmp31 - tmp32);
                    auto tmp34 = decltype(tmp23)(tmp23 + tmp30);
                    auto tmp35 = decltype(tmp33)(tmp33 * tmp29);
                    auto tmp36 = decltype(tmp32)(tmp32 + tmp35);
                    auto tmp37 = decltype(tmp34)(tmp34 - tmp36);

                    auto tmp38 = c10::convert<float>(tmp6);
                    auto tmp39 = decltype(tmp5)(tmp5 - tmp38);
                    auto tmp40 = max_propagate_nan(tmp39, tmp4);
                    auto tmp41 = min_propagate_nan(tmp40, tmp28);
                    auto tmp42 = decltype(tmp37)(tmp37 * tmp41);
                    auto tmp43 = decltype(tmp36)(tmp36 + tmp42);

                    in_out_ptr0[static_cast<long>(x2 + (ks2*x1) + (ks1*ks2*x0))] = tmp43;
                }
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
    buf1 = buf0; del buf0  # reuse
    buf3 = buf1; del buf1  # reuse
    buf4 = buf3; del buf3  # reuse
    cpp_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0(c_void_p(buf4.data_ptr()), c_void_p(arg3_1.data_ptr()), c_long(s0), c_long(s3), c_long(s4), c_long(s1), c_long(s2))
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
