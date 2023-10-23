
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


cpp_fused_clone_div_index_mul_sum_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/bf/cbf262yqjxhzxmw7lov36xiiezas3czyjs7cdvyrvlrje4xcl2kd.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr3)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(272L); x0+=static_cast<long>(1L))
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
            auto tmp19 = tmp14 < tmp18;
            auto tmp20 = tmp14 + tmp15;
            auto tmp21 = c10::convert<float>(tmp20);
            auto tmp22 = tmp21 - tmp5;
            auto tmp23 = tmp22 + tmp2;
            auto tmp24 = c10::convert<float>(0.5964912280701754);
            auto tmp25 = decltype(tmp23)(tmp23 * tmp24);
            auto tmp26 = std::abs(tmp25);
            auto tmp27 = c10::convert<float>(1.0);
            auto tmp28 = min_propagate_nan(tmp26, tmp27);
            auto tmp29 = tmp27 - tmp28;
            auto tmp30 = c10::convert<float>(0.0);
            auto tmp31 = tmp19 ? tmp29 : tmp30;
            auto tmp32 = c10::convert<long>(1);
            auto tmp33 = tmp32 < tmp18;
            auto tmp34 = tmp32 + tmp15;
            auto tmp35 = c10::convert<float>(tmp34);
            auto tmp36 = tmp35 - tmp5;
            auto tmp37 = tmp36 + tmp2;
            auto tmp38 = decltype(tmp37)(tmp37 * tmp24);
            auto tmp39 = std::abs(tmp38);
            auto tmp40 = min_propagate_nan(tmp39, tmp27);
            auto tmp41 = tmp27 - tmp40;
            auto tmp42 = tmp33 ? tmp41 : tmp30;
            auto tmp43 = tmp31 + tmp42;
            auto tmp44 = c10::convert<long>(2);
            auto tmp45 = tmp44 < tmp18;
            auto tmp46 = tmp44 + tmp15;
            auto tmp47 = c10::convert<float>(tmp46);
            auto tmp48 = tmp47 - tmp5;
            auto tmp49 = tmp48 + tmp2;
            auto tmp50 = decltype(tmp49)(tmp49 * tmp24);
            auto tmp51 = std::abs(tmp50);
            auto tmp52 = min_propagate_nan(tmp51, tmp27);
            auto tmp53 = tmp27 - tmp52;
            auto tmp54 = tmp45 ? tmp53 : tmp30;
            auto tmp55 = tmp43 + tmp54;
            auto tmp56 = c10::convert<long>(3);
            auto tmp57 = tmp56 < tmp18;
            auto tmp58 = tmp56 + tmp15;
            auto tmp59 = c10::convert<float>(tmp58);
            auto tmp60 = tmp59 - tmp5;
            auto tmp61 = tmp60 + tmp2;
            auto tmp62 = decltype(tmp61)(tmp61 * tmp24);
            auto tmp63 = std::abs(tmp62);
            auto tmp64 = min_propagate_nan(tmp63, tmp27);
            auto tmp65 = tmp27 - tmp64;
            auto tmp66 = tmp57 ? tmp65 : tmp30;
            auto tmp67 = tmp55 + tmp66;
            auto tmp68 = c10::convert<long>(4);
            auto tmp69 = tmp68 < tmp18;
            auto tmp70 = tmp68 + tmp15;
            auto tmp71 = c10::convert<float>(tmp70);
            auto tmp72 = tmp71 - tmp5;
            auto tmp73 = tmp72 + tmp2;
            auto tmp74 = decltype(tmp73)(tmp73 * tmp24);
            auto tmp75 = std::abs(tmp74);
            auto tmp76 = min_propagate_nan(tmp75, tmp27);
            auto tmp77 = tmp27 - tmp76;
            auto tmp78 = tmp69 ? tmp77 : tmp30;
            auto tmp79 = tmp67 + tmp78;
            out_ptr0[static_cast<long>(x0)] = tmp79;
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(5L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(272L); x1+=static_cast<long>(1L))
            {
                auto tmp33 = out_ptr0[static_cast<long>(x1)];
                auto tmp0 = c10::convert<long>(x1);
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
                auto tmp19 = c10::convert<long>(x0);
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
                auto tmp34 = tmp32 / tmp33;
                out_ptr1[static_cast<long>(x1 + (272L*x0))] = tmp34;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1380L); x0+=static_cast<long>(1L))
        {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(272L); x2+=static_cast<long>(1L))
                {

                                                                            auto tmp0 = out_ptr1[static_cast<long>(x2)];
                    auto tmp17 = out_ptr1[static_cast<long>(272L + x2)];
                    auto tmp24 = out_ptr1[static_cast<long>(544L + x2)];
                    auto tmp31 = out_ptr1[static_cast<long>(816L + x2)];
                    auto tmp38 = out_ptr1[static_cast<long>(1088L + x2)];
                    auto tmp1 = c10::convert<long>(x2);
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
                    auto tmp15 = in_ptr0[static_cast<long>(x1 + (3L*tmp14) + (1368L*x0))];
                    auto tmp16 = decltype(tmp0)(tmp0 * tmp15);
                    auto tmp18 = c10::convert<long>(1);
                    auto tmp19 = tmp11 + tmp18;

                    auto tmp20 = min_propagate_nan(tmp19, tmp13);
                    auto tmp21 = in_ptr0[static_cast<long>(x1 + (3L*tmp20) + (1368L*x0))];
                    auto tmp22 = decltype(tmp17)(tmp17 * tmp21);
                    auto tmp23 = tmp16 + tmp22;
                    auto tmp25 = c10::convert<long>(2);
                    auto tmp26 = tmp11 + tmp25;
                    auto tmp27 = min_propagate_nan(tmp26, tmp13);

                    auto tmp28 = in_ptr0[static_cast<long>(x1 + (3L*tmp27) + (1368L*x0))];
                    auto tmp29 = decltype(tmp24)(tmp24 * tmp28);
                    auto tmp30 = tmp23 + tmp29;
                    auto tmp32 = c10::convert<long>(3);
                    auto tmp33 = tmp11 + tmp32;
                    auto tmp34 = min_propagate_nan(tmp33, tmp13);

                    auto tmp35 = in_ptr0[static_cast<long>(x1 + (3L*tmp34) + (1368L*x0))];
                    auto tmp36 = decltype(tmp31)(tmp31 * tmp35);
                    auto tmp37 = tmp30 + tmp36;
                    auto tmp39 = c10::convert<long>(4);
                    auto tmp40 = tmp11 + tmp39;
                    auto tmp41 = min_propagate_nan(tmp40, tmp13);

                    auto tmp42 = in_ptr0[static_cast<long>(x1 + (3L*tmp41) + (1368L*x0))];
                    auto tmp43 = decltype(tmp38)(tmp38 * tmp42);
                    auto tmp44 = tmp37 + tmp43;
                    out_ptr3[static_cast<long>(x1 + (3L*x2) + (816L*x0))] = tmp44;
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
    assert_size_stride(arg0_1, (4, 3, 345, 456), (471960, 1, 1368, 3))
    buf1 = empty_strided((272, ), (1, ), device='cpu', dtype=torch.float32)
    buf2 = empty_strided((5, 272), (272, 1), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((4, 3, 345, 272), (281520, 1, 816, 3), device='cpu', dtype=torch.float32)
    cpp_fused_clone_div_index_mul_sum_0(c_void_p(arg0_1.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf2.data_ptr()), c_void_p(buf4.data_ptr()))
    del arg0_1
    return (buf4, )


def benchmark_compiled_module(times=5000, repeat=200):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4, 3, 345, 456), (471960, 1, 1368, 3), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
