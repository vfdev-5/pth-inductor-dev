
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
extern "C" void kernel(float* in_out_ptr1,
                       const float* in_ptr0,
                       const long ks0,
                       const long ks1,
                       const long ks2)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(1234L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(1345L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = c10::convert<long>(x1);
                    auto tmp1 = c10::convert<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = decltype(tmp1)(tmp1 + tmp2);
                    auto tmp4 = c10::convert<float>((1.0/1234.0)*ks1);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = decltype(tmp5)(tmp5 - tmp2);
                    auto tmp7 = static_cast<float>(0.0);
                    auto tmp8 = max_propagate_nan(tmp6, tmp7);
                    auto tmp9 = c10::convert<long>(tmp8);
                    auto tmp10 = c10::convert<long>((-1L) + ks1);
                    auto tmp11 = tmp9 < tmp10;
                    auto tmp12 = static_cast<long>(1);
                    auto tmp13 = decltype(tmp9)(tmp9 + tmp12);
                    auto tmp14 = tmp11 ? tmp13 : tmp9;
                    auto tmp15 = c10::convert<long>(x2);
                    auto tmp16 = c10::convert<float>(tmp15);
                    auto tmp17 = decltype(tmp16)(tmp16 + tmp2);
                    auto tmp18 = c10::convert<float>((1.0/1345.0)*ks2);
                    auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                    auto tmp20 = decltype(tmp19)(tmp19 - tmp2);
                    auto tmp21 = max_propagate_nan(tmp20, tmp7);
                    auto tmp22 = c10::convert<long>(tmp21);
                    auto tmp23 = c10::convert<long>((-1L) + ks2);
                    auto tmp24 = tmp22 < tmp23;
                    auto tmp25 = decltype(tmp22)(tmp22 + tmp12);
                    auto tmp26 = tmp24 ? tmp25 : tmp22;
                    auto tmp27 = in_ptr0[static_cast<long>(tmp26 + (ks2*tmp14) + (ks1*ks2*x0))];
                    auto tmp28 = in_ptr0[static_cast<long>(tmp22 + (ks2*tmp14) + (ks1*ks2*x0))];
                    auto tmp29 = in_ptr0[static_cast<long>(tmp26 + (ks2*tmp9) + (ks1*ks2*x0))];
                    auto tmp30 = in_ptr0[static_cast<long>(tmp22 + (ks2*tmp9) + (ks1*ks2*x0))];
                    auto tmp31 = decltype(tmp29)(tmp29 - tmp30);
                    auto tmp32 = c10::convert<float>(tmp22);
                    auto tmp33 = decltype(tmp21)(tmp21 - tmp32);
                    auto tmp34 = max_propagate_nan(tmp33, tmp7);
                    auto tmp35 = static_cast<float>(1.0);
                    auto tmp36 = min_propagate_nan(tmp34, tmp35);
                    auto tmp37 = decltype(tmp31)(tmp31 * tmp36);
                    auto tmp38 = decltype(tmp27)(tmp27 - tmp28);
                    auto tmp39 = decltype(tmp38)(tmp38 * tmp36);
                    auto tmp40 = decltype(tmp28)(tmp28 + tmp39);
                    auto tmp41 = decltype(tmp30)(tmp30 + tmp37);
                    auto tmp42 = decltype(tmp40)(tmp40 - tmp41);
                    auto tmp43 = c10::convert<float>(tmp9);
                    auto tmp44 = decltype(tmp8)(tmp8 - tmp43);
                    auto tmp45 = max_propagate_nan(tmp44, tmp7);
                    auto tmp46 = min_propagate_nan(tmp45, tmp35);
                    auto tmp47 = decltype(tmp42)(tmp42 * tmp46);
                    auto tmp48 = decltype(tmp41)(tmp41 + tmp47);
                    in_out_ptr1[static_cast<long>(x2 + (1345L*x1) + (1659730L*x0))] = tmp48;
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
    assert_size_stride(arg3_1, (1, s0, s1, s2), (s0*s1*s2, s1*s2, s2, 1))
    buf0 = empty((1, s0, 1234, 1345), device='cpu', dtype=torch.float32)
    buf4 = buf0; del buf0  # reuse
    buf5 = buf4; del buf4  # reuse
    cpp_fused__to_copy__unsafe_index_add_arange_clamp_mul_sub_0(c_void_p(buf5.data_ptr()), c_void_p(arg3_1.data_ptr()), c_long(s0), c_long(s1), c_long(s2))
    del arg3_1
    return (buf5, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 3
    arg1_1 = 2345
    arg2_1 = 2456
    arg3_1 = rand_strided((1, 3, 2345, 2456), (17277960, 5759320, 2456, 1), device='cpu', dtype=torch.float32)
    arg4_1 = 1234
    arg5_1 = 1345
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
