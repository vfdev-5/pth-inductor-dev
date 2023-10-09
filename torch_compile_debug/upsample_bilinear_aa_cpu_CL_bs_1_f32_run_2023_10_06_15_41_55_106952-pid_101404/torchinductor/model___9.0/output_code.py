
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


cpp_fused_add_clone_div_index_mul_sum_unbind_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr4,
                       float* out_ptr10)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(272L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = static_cast<long>(x0);
            auto tmp1 = static_cast<float>(tmp0);
            auto tmp2 = static_cast<float>(0.5);
            auto tmp3 = tmp1 + tmp2;
            auto tmp4 = static_cast<float>(1.6764705882352942);
            auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
            auto tmp6 = tmp5 + tmp4;
            auto tmp7 = tmp6 + tmp2;
            auto tmp8 = static_cast<long>(tmp7);
            auto tmp9 = static_cast<long>(456);
            auto tmp10 = min_propagate_nan(tmp8, tmp9);
            auto tmp11 = tmp5 - tmp4;
            auto tmp12 = tmp11 + tmp2;
            auto tmp13 = static_cast<long>(tmp12);
            auto tmp14 = static_cast<long>(0);
            auto tmp15 = max_propagate_nan(tmp13, tmp14);
            auto tmp16 = tmp10 - tmp15;
            auto tmp17 = static_cast<long>(5);
            auto tmp18 = min_propagate_nan(tmp16, tmp17);
            auto tmp19 = tmp14 < tmp18;
            auto tmp20 = tmp14 + tmp15;
            auto tmp21 = static_cast<float>(tmp20);
            auto tmp22 = tmp21 - tmp5;
            auto tmp23 = tmp22 + tmp2;
            auto tmp24 = static_cast<float>(0.5964912280701754);
            auto tmp25 = decltype(tmp23)(tmp23 * tmp24);
            auto tmp26 = std::abs(tmp25);
            auto tmp27 = static_cast<float>(1.0);
            auto tmp28 = min_propagate_nan(tmp26, tmp27);
            auto tmp29 = tmp27 - tmp28;
            auto tmp30 = static_cast<float>(0.0);
            auto tmp31 = tmp19 ? tmp29 : tmp30;
            auto tmp32 = static_cast<long>(1);
            auto tmp33 = tmp32 < tmp18;
            auto tmp34 = tmp32 + tmp15;
            auto tmp35 = static_cast<float>(tmp34);
            auto tmp36 = tmp35 - tmp5;
            auto tmp37 = tmp36 + tmp2;
            auto tmp38 = decltype(tmp37)(tmp37 * tmp24);
            auto tmp39 = std::abs(tmp38);
            auto tmp40 = min_propagate_nan(tmp39, tmp27);
            auto tmp41 = tmp27 - tmp40;
            auto tmp42 = tmp33 ? tmp41 : tmp30;
            auto tmp43 = tmp31 + tmp42;
            auto tmp44 = static_cast<long>(2);
            auto tmp45 = tmp44 < tmp18;
            auto tmp46 = tmp44 + tmp15;
            auto tmp47 = static_cast<float>(tmp46);
            auto tmp48 = tmp47 - tmp5;
            auto tmp49 = tmp48 + tmp2;
            auto tmp50 = decltype(tmp49)(tmp49 * tmp24);
            auto tmp51 = std::abs(tmp50);
            auto tmp52 = min_propagate_nan(tmp51, tmp27);
            auto tmp53 = tmp27 - tmp52;
            auto tmp54 = tmp45 ? tmp53 : tmp30;
            auto tmp55 = tmp43 + tmp54;
            auto tmp56 = static_cast<long>(3);
            auto tmp57 = tmp56 < tmp18;
            auto tmp58 = tmp56 + tmp15;
            auto tmp59 = static_cast<float>(tmp58);
            auto tmp60 = tmp59 - tmp5;
            auto tmp61 = tmp60 + tmp2;
            auto tmp62 = decltype(tmp61)(tmp61 * tmp24);
            auto tmp63 = std::abs(tmp62);
            auto tmp64 = min_propagate_nan(tmp63, tmp27);
            auto tmp65 = tmp27 - tmp64;
            auto tmp66 = tmp57 ? tmp65 : tmp30;
            auto tmp67 = tmp55 + tmp66;
            auto tmp68 = static_cast<long>(4);
            auto tmp69 = tmp68 < tmp18;
            auto tmp70 = tmp68 + tmp15;
            auto tmp71 = static_cast<float>(tmp70);
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(271L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = static_cast<long>(x0);
            auto tmp1 = static_cast<float>(tmp0);
            auto tmp2 = static_cast<float>(0.5);
            auto tmp3 = tmp1 + tmp2;
            auto tmp4 = static_cast<float>(1.2730627306273063);
            auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
            auto tmp6 = tmp5 + tmp4;
            auto tmp7 = tmp6 + tmp2;
            auto tmp8 = static_cast<long>(tmp7);
            auto tmp9 = static_cast<long>(345);
            auto tmp10 = min_propagate_nan(tmp8, tmp9);
            auto tmp11 = tmp5 - tmp4;
            auto tmp12 = tmp11 + tmp2;
            auto tmp13 = static_cast<long>(tmp12);
            auto tmp14 = static_cast<long>(0);
            auto tmp15 = max_propagate_nan(tmp13, tmp14);
            auto tmp16 = tmp10 - tmp15;
            auto tmp17 = static_cast<long>(5);
            auto tmp18 = min_propagate_nan(tmp16, tmp17);
            auto tmp19 = tmp14 < tmp18;
            auto tmp20 = tmp14 + tmp15;
            auto tmp21 = static_cast<float>(tmp20);
            auto tmp22 = tmp21 - tmp5;
            auto tmp23 = tmp22 + tmp2;
            auto tmp24 = static_cast<float>(0.7855072463768116);
            auto tmp25 = decltype(tmp23)(tmp23 * tmp24);
            auto tmp26 = std::abs(tmp25);
            auto tmp27 = static_cast<float>(1.0);
            auto tmp28 = min_propagate_nan(tmp26, tmp27);
            auto tmp29 = tmp27 - tmp28;
            auto tmp30 = static_cast<float>(0.0);
            auto tmp31 = tmp19 ? tmp29 : tmp30;
            auto tmp32 = static_cast<long>(1);
            auto tmp33 = tmp32 < tmp18;
            auto tmp34 = tmp32 + tmp15;
            auto tmp35 = static_cast<float>(tmp34);
            auto tmp36 = tmp35 - tmp5;
            auto tmp37 = tmp36 + tmp2;
            auto tmp38 = decltype(tmp37)(tmp37 * tmp24);
            auto tmp39 = std::abs(tmp38);
            auto tmp40 = min_propagate_nan(tmp39, tmp27);
            auto tmp41 = tmp27 - tmp40;
            auto tmp42 = tmp33 ? tmp41 : tmp30;
            auto tmp43 = tmp31 + tmp42;
            auto tmp44 = static_cast<long>(2);
            auto tmp45 = tmp44 < tmp18;
            auto tmp46 = tmp44 + tmp15;
            auto tmp47 = static_cast<float>(tmp46);
            auto tmp48 = tmp47 - tmp5;
            auto tmp49 = tmp48 + tmp2;
            auto tmp50 = decltype(tmp49)(tmp49 * tmp24);
            auto tmp51 = std::abs(tmp50);
            auto tmp52 = min_propagate_nan(tmp51, tmp27);
            auto tmp53 = tmp27 - tmp52;
            auto tmp54 = tmp45 ? tmp53 : tmp30;
            auto tmp55 = tmp43 + tmp54;
            auto tmp56 = static_cast<long>(3);
            auto tmp57 = tmp56 < tmp18;
            auto tmp58 = tmp56 + tmp15;
            auto tmp59 = static_cast<float>(tmp58);
            auto tmp60 = tmp59 - tmp5;
            auto tmp61 = tmp60 + tmp2;
            auto tmp62 = decltype(tmp61)(tmp61 * tmp24);
            auto tmp63 = std::abs(tmp62);
            auto tmp64 = min_propagate_nan(tmp63, tmp27);
            auto tmp65 = tmp27 - tmp64;
            auto tmp66 = tmp57 ? tmp65 : tmp30;
            auto tmp67 = tmp55 + tmp66;
            auto tmp68 = static_cast<long>(4);
            auto tmp69 = tmp68 < tmp18;
            auto tmp70 = tmp68 + tmp15;
            auto tmp71 = static_cast<float>(tmp70);
            auto tmp72 = tmp71 - tmp5;
            auto tmp73 = tmp72 + tmp2;
            auto tmp74 = decltype(tmp73)(tmp73 * tmp24);
            auto tmp75 = std::abs(tmp74);
            auto tmp76 = min_propagate_nan(tmp75, tmp27);
            auto tmp77 = tmp27 - tmp76;
            auto tmp78 = tmp69 ? tmp77 : tmp30;
            auto tmp79 = tmp67 + tmp78;
            out_ptr1[static_cast<long>(x0)] = tmp79;
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
                auto tmp0 = static_cast<long>(x1);
                auto tmp1 = static_cast<float>(tmp0);
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = tmp1 + tmp2;
                auto tmp4 = static_cast<float>(1.6764705882352942);
                auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                auto tmp6 = tmp5 + tmp4;
                auto tmp7 = tmp6 + tmp2;
                auto tmp8 = static_cast<long>(tmp7);
                auto tmp9 = static_cast<long>(456);
                auto tmp10 = min_propagate_nan(tmp8, tmp9);
                auto tmp11 = tmp5 - tmp4;
                auto tmp12 = tmp11 + tmp2;
                auto tmp13 = static_cast<long>(tmp12);
                auto tmp14 = static_cast<long>(0);
                auto tmp15 = max_propagate_nan(tmp13, tmp14);
                auto tmp16 = tmp10 - tmp15;
                auto tmp17 = static_cast<long>(5);
                auto tmp18 = min_propagate_nan(tmp16, tmp17);
                auto tmp19 = static_cast<long>(x0);
                auto tmp20 = tmp19 < tmp18;
                auto tmp21 = tmp19 + tmp15;
                auto tmp22 = static_cast<float>(tmp21);
                auto tmp23 = tmp22 - tmp5;
                auto tmp24 = tmp23 + tmp2;
                auto tmp25 = static_cast<float>(0.5964912280701754);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp27 = std::abs(tmp26);
                auto tmp28 = static_cast<float>(1.0);
                auto tmp29 = min_propagate_nan(tmp27, tmp28);
                auto tmp30 = tmp28 - tmp29;
                auto tmp31 = static_cast<float>(0.0);
                auto tmp32 = tmp20 ? tmp30 : tmp31;
                auto tmp34 = tmp32 / tmp33;
                out_ptr2[static_cast<long>(x1 + (272L*x0))] = tmp34;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(345L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(272L); x2+=static_cast<long>(1L))
                {
                    auto tmp15 = out_ptr2[static_cast<long>(x2)];
                    auto tmp21 = out_ptr2[static_cast<long>(272L + x2)];
                    auto tmp28 = out_ptr2[static_cast<long>(544L + x2)];
                    auto tmp35 = out_ptr2[static_cast<long>(816L + x2)];
                    auto tmp0 = static_cast<long>(x2);
                    auto tmp1 = static_cast<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(1.6764705882352942);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = tmp5 - tmp4;
                    auto tmp7 = tmp6 + tmp2;
                    auto tmp8 = static_cast<long>(tmp7);
                    auto tmp9 = static_cast<long>(0);
                    auto tmp10 = max_propagate_nan(tmp8, tmp9);
                    auto tmp11 = tmp10 + tmp9;
                    auto tmp12 = static_cast<long>(455);
                    auto tmp13 = min_propagate_nan(tmp11, tmp12);
                    auto tmp14 = in_ptr0[static_cast<long>(x0 + (3L*tmp13) + (1368L*x1))];
                    auto tmp16 = decltype(tmp14)(tmp14 * tmp15);
                    auto tmp17 = static_cast<long>(1);
                    auto tmp18 = tmp10 + tmp17;
                    auto tmp19 = min_propagate_nan(tmp18, tmp12);
                    auto tmp20 = in_ptr0[static_cast<long>(x0 + (3L*tmp19) + (1368L*x1))];
                    auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                    auto tmp23 = tmp16 + tmp22;
                    auto tmp24 = static_cast<long>(2);
                    auto tmp25 = tmp10 + tmp24;
                    auto tmp26 = min_propagate_nan(tmp25, tmp12);
                    auto tmp27 = in_ptr0[static_cast<long>(x0 + (3L*tmp26) + (1368L*x1))];
                    auto tmp29 = decltype(tmp27)(tmp27 * tmp28);
                    auto tmp30 = tmp23 + tmp29;
                    auto tmp31 = static_cast<long>(3);
                    auto tmp32 = tmp10 + tmp31;
                    auto tmp33 = min_propagate_nan(tmp32, tmp12);
                    auto tmp34 = in_ptr0[static_cast<long>(x0 + (3L*tmp33) + (1368L*x1))];
                    auto tmp36 = decltype(tmp34)(tmp34 * tmp35);
                    auto tmp37 = tmp30 + tmp36;
                    in_out_ptr0[static_cast<long>(x2 + (272L*x1) + (93840L*x0))] = tmp37;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(5L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(271L); x1+=static_cast<long>(1L))
            {
                auto tmp33 = out_ptr1[static_cast<long>(x1)];
                auto tmp0 = static_cast<long>(x1);
                auto tmp1 = static_cast<float>(tmp0);
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = tmp1 + tmp2;
                auto tmp4 = static_cast<float>(1.2730627306273063);
                auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                auto tmp6 = tmp5 + tmp4;
                auto tmp7 = tmp6 + tmp2;
                auto tmp8 = static_cast<long>(tmp7);
                auto tmp9 = static_cast<long>(345);
                auto tmp10 = min_propagate_nan(tmp8, tmp9);
                auto tmp11 = tmp5 - tmp4;
                auto tmp12 = tmp11 + tmp2;
                auto tmp13 = static_cast<long>(tmp12);
                auto tmp14 = static_cast<long>(0);
                auto tmp15 = max_propagate_nan(tmp13, tmp14);
                auto tmp16 = tmp10 - tmp15;
                auto tmp17 = static_cast<long>(5);
                auto tmp18 = min_propagate_nan(tmp16, tmp17);
                auto tmp19 = static_cast<long>(x0);
                auto tmp20 = tmp19 < tmp18;
                auto tmp21 = tmp19 + tmp15;
                auto tmp22 = static_cast<float>(tmp21);
                auto tmp23 = tmp22 - tmp5;
                auto tmp24 = tmp23 + tmp2;
                auto tmp25 = static_cast<float>(0.7855072463768116);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp27 = std::abs(tmp26);
                auto tmp28 = static_cast<float>(1.0);
                auto tmp29 = min_propagate_nan(tmp27, tmp28);
                auto tmp30 = tmp28 - tmp29;
                auto tmp31 = static_cast<float>(0.0);
                auto tmp32 = tmp20 ? tmp30 : tmp31;
                auto tmp34 = tmp32 / tmp33;
                out_ptr4[static_cast<long>(x1 + (271L*x0))] = tmp34;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(271L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(272L); x2+=static_cast<long>(1L))
                {
                    auto tmp59 = out_ptr4[static_cast<long>(x1)];
                    auto tmp61 = out_ptr4[static_cast<long>(271L + x1)];
                    auto tmp64 = out_ptr4[static_cast<long>(542L + x1)];
                    auto tmp67 = out_ptr4[static_cast<long>(813L + x1)];
                    auto tmp70 = out_ptr4[static_cast<long>(1084L + x1)];
                    auto tmp0 = static_cast<long>(x1);
                    auto tmp1 = static_cast<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(1.2730627306273063);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = tmp5 - tmp4;
                    auto tmp7 = tmp6 + tmp2;
                    auto tmp8 = static_cast<long>(tmp7);
                    auto tmp9 = static_cast<long>(0);
                    auto tmp10 = max_propagate_nan(tmp8, tmp9);
                    auto tmp11 = tmp10 + tmp9;
                    auto tmp12 = static_cast<long>(344);
                    auto tmp13 = min_propagate_nan(tmp11, tmp12);
                    auto tmp14 = in_out_ptr0[static_cast<long>(x2 + (272L*tmp13) + (93840L*x0))];
                    auto tmp15 = static_cast<long>(x2);
                    auto tmp16 = static_cast<float>(tmp15);
                    auto tmp17 = tmp16 + tmp2;
                    auto tmp18 = static_cast<float>(1.6764705882352942);
                    auto tmp19 = decltype(tmp17)(tmp17 * tmp18);
                    auto tmp20 = tmp19 - tmp18;
                    auto tmp21 = tmp20 + tmp2;
                    auto tmp22 = static_cast<long>(tmp21);
                    auto tmp23 = max_propagate_nan(tmp22, tmp9);
                    auto tmp24 = static_cast<long>(4);
                    auto tmp25 = tmp23 + tmp24;
                    auto tmp26 = static_cast<long>(455);
                    auto tmp27 = min_propagate_nan(tmp25, tmp26);
                    auto tmp28 = in_ptr0[static_cast<long>(x0 + (3L*tmp27) + (1368L*tmp13))];
                    auto tmp29 = static_cast<float>(0.0);
                    auto tmp30 = decltype(tmp28)(tmp28 * tmp29);
                    auto tmp31 = tmp14 + tmp30;
                    auto tmp32 = static_cast<long>(1);
                    auto tmp33 = tmp10 + tmp32;
                    auto tmp34 = min_propagate_nan(tmp33, tmp12);
                    auto tmp35 = in_out_ptr0[static_cast<long>(x2 + (272L*tmp34) + (93840L*x0))];
                    auto tmp36 = in_ptr0[static_cast<long>(x0 + (3L*tmp27) + (1368L*tmp34))];
                    auto tmp37 = decltype(tmp36)(tmp36 * tmp29);
                    auto tmp38 = tmp35 + tmp37;
                    auto tmp39 = static_cast<long>(2);
                    auto tmp40 = tmp10 + tmp39;
                    auto tmp41 = min_propagate_nan(tmp40, tmp12);
                    auto tmp42 = in_out_ptr0[static_cast<long>(x2 + (272L*tmp41) + (93840L*x0))];
                    auto tmp43 = in_ptr0[static_cast<long>(x0 + (3L*tmp27) + (1368L*tmp41))];
                    auto tmp44 = decltype(tmp43)(tmp43 * tmp29);
                    auto tmp45 = tmp42 + tmp44;
                    auto tmp46 = static_cast<long>(3);
                    auto tmp47 = tmp10 + tmp46;
                    auto tmp48 = min_propagate_nan(tmp47, tmp12);
                    auto tmp49 = in_out_ptr0[static_cast<long>(x2 + (272L*tmp48) + (93840L*x0))];
                    auto tmp50 = in_ptr0[static_cast<long>(x0 + (3L*tmp27) + (1368L*tmp48))];
                    auto tmp51 = decltype(tmp50)(tmp50 * tmp29);
                    auto tmp52 = tmp49 + tmp51;
                    auto tmp53 = tmp10 + tmp24;
                    auto tmp54 = min_propagate_nan(tmp53, tmp12);
                    auto tmp55 = in_out_ptr0[static_cast<long>(x2 + (272L*tmp54) + (93840L*x0))];
                    auto tmp56 = in_ptr0[static_cast<long>(x0 + (3L*tmp27) + (1368L*tmp54))];
                    auto tmp57 = decltype(tmp56)(tmp56 * tmp29);
                    auto tmp58 = tmp55 + tmp57;
                    auto tmp60 = decltype(tmp31)(tmp31 * tmp59);
                    auto tmp62 = decltype(tmp38)(tmp38 * tmp61);
                    auto tmp63 = tmp60 + tmp62;
                    auto tmp65 = decltype(tmp45)(tmp45 * tmp64);
                    auto tmp66 = tmp63 + tmp65;
                    auto tmp68 = decltype(tmp52)(tmp52 * tmp67);
                    auto tmp69 = tmp66 + tmp68;
                    auto tmp71 = decltype(tmp58)(tmp58 * tmp70);
                    auto tmp72 = tmp69 + tmp71;
                    in_out_ptr1[static_cast<long>(x2 + (272L*x1) + (73712L*x0))] = tmp72;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(73712L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (73712L*x0))];
                out_ptr10[static_cast<long>(x0 + (3L*x1))] = tmp0;
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
    assert_size_stride(arg0_1, (1, 3, 345, 456), (471960, 1, 1368, 3))
    buf1 = empty_strided((272, ), (1, ), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((271, ), (1, ), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((5, 272), (272, 1), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((1, 3, 345, 272), (281520, 93840, 272, 1), device='cpu', dtype=torch.float32)
    buf6 = buf5; del buf5  # reuse
    buf8 = empty_strided((5, 271), (271, 1), device='cpu', dtype=torch.float32)
    buf14 = empty_strided((1, 3, 271, 272), (221136, 1, 816, 3), device='cpu', dtype=torch.float32)
    buf13 = buf10; del buf10  # reuse
    buf14 = empty_strided((1, 3, 271, 272), (221136, 1, 816, 3), device='cpu', dtype=torch.float32)
    cpp_fused_add_clone_div_index_mul_sum_unbind_0(c_void_p(buf6.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf14.data_ptr()))
    del arg0_1
    return (buf14, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 3, 345, 456), (471960, 1, 1368, 3), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
