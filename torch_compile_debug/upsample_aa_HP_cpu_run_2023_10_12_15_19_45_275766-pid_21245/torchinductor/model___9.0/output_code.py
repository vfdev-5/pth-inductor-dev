
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


cpp_fused_div_index_mul_sum_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/bf/cbf262yqjxhzxmw7lov36xiiezas3czyjs7cdvyrvlrje4xcl2kd.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(272L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(5L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = c10::convert<long>(x0);
                auto tmp1 = c10::convert<float>(tmp0);
                auto tmp2 = c10::convert<float>(0.5);
                auto tmp3 = tmp1 + tmp2;
                auto tmp4 = c10::convert<float>(1.6764705882352942);
                auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                auto tmp6 = tmp5 + tmp4;
                auto tmp7 = tmp6 + tmp2;
                auto tmp8 = c10::convert<long>(tmp7);
                auto tmp9 = c10::convert<long>(456);
                auto tmp10 = min_propagate_nan(tmp8, tmp9);
                auto tmp11 = tmp5 - tmp4;
                auto tmp12 = tmp11 + tmp2;
                auto tmp13 = c10::convert<long>(tmp12);
                auto tmp14 = c10::convert<long>(0);
                auto tmp15 = max_propagate_nan(tmp13, tmp14);
                auto tmp16 = tmp10 - tmp15;
                auto tmp17 = c10::convert<long>(5);
                auto tmp18 = min_propagate_nan(tmp16, tmp17);
                auto tmp19 = c10::convert<long>(x1);
                auto tmp20 = tmp19 < tmp18;
                auto tmp21 = tmp19 + tmp15;
                auto tmp22 = c10::convert<float>(tmp21);
                auto tmp23 = tmp22 - tmp5;
                auto tmp24 = tmp23 + tmp2;
                auto tmp25 = c10::convert<float>(0.5964912280701754);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp27 = std::abs(tmp26);
                auto tmp28 = c10::convert<float>(1.0);
                auto tmp29 = min_propagate_nan(tmp27, tmp28);
                auto tmp30 = tmp28 - tmp29;
                auto tmp31 = c10::convert<float>(0.0);
                auto tmp32 = tmp20 ? tmp30 : tmp31;
                auto tmp33 = tmp14 < tmp18;
                auto tmp34 = tmp14 + tmp15;
                auto tmp35 = c10::convert<float>(tmp34);
                auto tmp36 = tmp35 - tmp5;
                auto tmp37 = tmp36 + tmp2;
                auto tmp38 = decltype(tmp37)(tmp37 * tmp25);
                auto tmp39 = std::abs(tmp38);
                auto tmp40 = min_propagate_nan(tmp39, tmp28);
                auto tmp41 = tmp28 - tmp40;
                auto tmp42 = tmp33 ? tmp41 : tmp31;
                auto tmp43 = c10::convert<long>(1);
                auto tmp44 = tmp43 < tmp18;
                auto tmp45 = tmp43 + tmp15;
                auto tmp46 = c10::convert<float>(tmp45);
                auto tmp47 = tmp46 - tmp5;
                auto tmp48 = tmp47 + tmp2;
                auto tmp49 = decltype(tmp48)(tmp48 * tmp25);
                auto tmp50 = std::abs(tmp49);
                auto tmp51 = min_propagate_nan(tmp50, tmp28);
                auto tmp52 = tmp28 - tmp51;
                auto tmp53 = tmp44 ? tmp52 : tmp31;
                auto tmp54 = tmp42 + tmp53;
                auto tmp55 = c10::convert<long>(2);
                auto tmp56 = tmp55 < tmp18;
                auto tmp57 = tmp55 + tmp15;
                auto tmp58 = c10::convert<float>(tmp57);
                auto tmp59 = tmp58 - tmp5;
                auto tmp60 = tmp59 + tmp2;
                auto tmp61 = decltype(tmp60)(tmp60 * tmp25);
                auto tmp62 = std::abs(tmp61);
                auto tmp63 = min_propagate_nan(tmp62, tmp28);
                auto tmp64 = tmp28 - tmp63;
                auto tmp65 = tmp56 ? tmp64 : tmp31;
                auto tmp66 = tmp54 + tmp65;
                auto tmp67 = c10::convert<long>(3);
                auto tmp68 = tmp67 < tmp18;
                auto tmp69 = tmp67 + tmp15;
                auto tmp70 = c10::convert<float>(tmp69);
                auto tmp71 = tmp70 - tmp5;
                auto tmp72 = tmp71 + tmp2;
                auto tmp73 = decltype(tmp72)(tmp72 * tmp25);
                auto tmp74 = std::abs(tmp73);
                auto tmp75 = min_propagate_nan(tmp74, tmp28);
                auto tmp76 = tmp28 - tmp75;
                auto tmp77 = tmp68 ? tmp76 : tmp31;
                auto tmp78 = tmp66 + tmp77;
                auto tmp79 = c10::convert<long>(4);
                auto tmp80 = tmp79 < tmp18;
                auto tmp81 = tmp79 + tmp15;
                auto tmp82 = c10::convert<float>(tmp81);
                auto tmp83 = tmp82 - tmp5;
                auto tmp84 = tmp83 + tmp2;
                auto tmp85 = decltype(tmp84)(tmp84 * tmp25);
                auto tmp86 = std::abs(tmp85);
                auto tmp87 = min_propagate_nan(tmp86, tmp28);
                auto tmp88 = tmp28 - tmp87;
                auto tmp89 = tmp80 ? tmp88 : tmp31;
                auto tmp90 = tmp78 + tmp89;
                auto tmp91 = tmp32 / tmp90;
                out_ptr0[static_cast<long>(x1 + (5L*x0))] = tmp91;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4140L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(272L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr0[static_cast<long>(5L*x1)];
                auto tmp17 = out_ptr0[static_cast<long>(1L + (5L*x1))];
                auto tmp24 = out_ptr0[static_cast<long>(2L + (5L*x1))];
                auto tmp31 = out_ptr0[static_cast<long>(3L + (5L*x1))];
                auto tmp38 = out_ptr0[static_cast<long>(4L + (5L*x1))];
                auto tmp1 = c10::convert<long>(x1);
                auto tmp2 = c10::convert<float>(tmp1);
                auto tmp3 = c10::convert<float>(0.5);
                auto tmp4 = tmp2 + tmp3;
                auto tmp5 = c10::convert<float>(1.6764705882352942);
                auto tmp6 = decltype(tmp4)(tmp4 * tmp5);
                auto tmp7 = tmp6 - tmp5;
                auto tmp8 = tmp7 + tmp3;
                auto tmp9 = c10::convert<long>(tmp8);
                auto tmp10 = c10::convert<long>(0);
                auto tmp11 = max_propagate_nan(tmp9, tmp10);
                auto tmp12 = tmp11 + tmp10;
                auto tmp13 = c10::convert<long>(455);
                auto tmp14 = min_propagate_nan(tmp12, tmp13);
                auto tmp15 = in_ptr0[static_cast<long>(tmp14 + (456L*x0))];
                auto tmp16 = decltype(tmp0)(tmp0 * tmp15);
                auto tmp18 = c10::convert<long>(1);
                auto tmp19 = tmp11 + tmp18;
                auto tmp20 = min_propagate_nan(tmp19, tmp13);
                auto tmp21 = in_ptr0[static_cast<long>(tmp20 + (456L*x0))];
                auto tmp22 = decltype(tmp17)(tmp17 * tmp21);
                auto tmp23 = tmp16 + tmp22;
                auto tmp25 = c10::convert<long>(2);
                auto tmp26 = tmp11 + tmp25;
                auto tmp27 = min_propagate_nan(tmp26, tmp13);
                auto tmp28 = in_ptr0[static_cast<long>(tmp27 + (456L*x0))];
                auto tmp29 = decltype(tmp24)(tmp24 * tmp28);
                auto tmp30 = tmp23 + tmp29;
                auto tmp32 = c10::convert<long>(3);
                auto tmp33 = tmp11 + tmp32;
                auto tmp34 = min_propagate_nan(tmp33, tmp13);
                auto tmp35 = in_ptr0[static_cast<long>(tmp34 + (456L*x0))];
                auto tmp36 = decltype(tmp31)(tmp31 * tmp35);
                auto tmp37 = tmp30 + tmp36;
                auto tmp39 = c10::convert<long>(4);
                auto tmp40 = tmp11 + tmp39;
                auto tmp41 = min_propagate_nan(tmp40, tmp13);
                auto tmp42 = in_ptr0[static_cast<long>(tmp41 + (456L*x0))];
                auto tmp43 = decltype(tmp38)(tmp38 * tmp42);
                auto tmp44 = tmp37 + tmp43;
                out_ptr1[static_cast<long>(x1 + (272L*x0))] = tmp44;
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
    assert_size_stride(arg0_1, (4, 3, 345, 456), (471960, 157320, 456, 1))
    buf2 = empty_strided((272, 5), (5, 1), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((4, 3, 345, 272), (281520, 93840, 272, 1), device='cpu', dtype=torch.float32)
    cpp_fused_div_index_mul_sum_0(c_void_p(arg0_1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf3.data_ptr()))
    del arg0_1
    return (buf3, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 3, 345, 456), (471960, 157320, 456, 1), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
