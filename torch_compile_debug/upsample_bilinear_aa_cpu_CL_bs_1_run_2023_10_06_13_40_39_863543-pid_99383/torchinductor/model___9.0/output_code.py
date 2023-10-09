
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


cpp_fused__to_copy_add_clone_div_index_mul_round_sum_unbind_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/ib/cibrnuq56cxamjj4krp4zpjvsirbmlolpbnmomodzyd46huzhdw7.h"
extern "C" void kernel(float* in_out_ptr0,
                       float* in_out_ptr1,
                       const unsigned char* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr4,
                       unsigned char* out_ptr10)
{
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(270L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = static_cast<long>(x0);
            auto tmp1 = static_cast<float>(tmp0);
            auto tmp2 = static_cast<float>(0.5);
            auto tmp3 = tmp1 + tmp2;
            auto tmp4 = static_cast<float>(1.6888888888888889);
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
            auto tmp24 = static_cast<float>(0.5921052631578947);
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
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(270L); x0+=static_cast<long>(1L))
        {
            auto tmp0 = static_cast<long>(x0);
            auto tmp1 = static_cast<float>(tmp0);
            auto tmp2 = static_cast<float>(0.5);
            auto tmp3 = tmp1 + tmp2;
            auto tmp4 = static_cast<float>(1.2777777777777777);
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
            auto tmp24 = static_cast<float>(0.782608695652174);
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
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(270L); x1+=static_cast<long>(1L))
            {
                auto tmp33 = out_ptr0[static_cast<long>(x1)];
                auto tmp0 = static_cast<long>(x1);
                auto tmp1 = static_cast<float>(tmp0);
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = tmp1 + tmp2;
                auto tmp4 = static_cast<float>(1.6888888888888889);
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
                auto tmp25 = static_cast<float>(0.5921052631578947);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp27 = std::abs(tmp26);
                auto tmp28 = static_cast<float>(1.0);
                auto tmp29 = min_propagate_nan(tmp27, tmp28);
                auto tmp30 = tmp28 - tmp29;
                auto tmp31 = static_cast<float>(0.0);
                auto tmp32 = tmp20 ? tmp30 : tmp31;
                auto tmp34 = tmp32 / tmp33;
                out_ptr2[static_cast<long>(x1 + (270L*x0))] = tmp34;
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
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(270L); x2+=static_cast<long>(1L))
                {
                    auto tmp16 = out_ptr2[static_cast<long>(x2)];
                    auto tmp23 = out_ptr2[static_cast<long>(270L + x2)];
                    auto tmp31 = out_ptr2[static_cast<long>(540L + x2)];
                    auto tmp39 = out_ptr2[static_cast<long>(810L + x2)];
                    auto tmp0 = static_cast<long>(x2);
                    auto tmp1 = static_cast<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(1.6888888888888889);
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
                    auto tmp15 = static_cast<float>(tmp14);
                    auto tmp17 = decltype(tmp15)(tmp15 * tmp16);
                    auto tmp18 = static_cast<long>(1);
                    auto tmp19 = tmp10 + tmp18;
                    auto tmp20 = min_propagate_nan(tmp19, tmp12);
                    auto tmp21 = in_ptr0[static_cast<long>(x0 + (3L*tmp20) + (1368L*x1))];
                    auto tmp22 = static_cast<float>(tmp21);
                    auto tmp24 = decltype(tmp22)(tmp22 * tmp23);
                    auto tmp25 = tmp17 + tmp24;
                    auto tmp26 = static_cast<long>(2);
                    auto tmp27 = tmp10 + tmp26;
                    auto tmp28 = min_propagate_nan(tmp27, tmp12);
                    auto tmp29 = in_ptr0[static_cast<long>(x0 + (3L*tmp28) + (1368L*x1))];
                    auto tmp30 = static_cast<float>(tmp29);
                    auto tmp32 = decltype(tmp30)(tmp30 * tmp31);
                    auto tmp33 = tmp25 + tmp32;
                    auto tmp34 = static_cast<long>(3);
                    auto tmp35 = tmp10 + tmp34;
                    auto tmp36 = min_propagate_nan(tmp35, tmp12);
                    auto tmp37 = in_ptr0[static_cast<long>(x0 + (3L*tmp36) + (1368L*x1))];
                    auto tmp38 = static_cast<float>(tmp37);
                    auto tmp40 = decltype(tmp38)(tmp38 * tmp39);
                    auto tmp41 = tmp33 + tmp40;
                    in_out_ptr0[static_cast<long>(x2 + (270L*x1) + (93150L*x0))] = tmp41;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(5L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(270L); x1+=static_cast<long>(1L))
            {
                auto tmp33 = out_ptr1[static_cast<long>(x1)];
                auto tmp0 = static_cast<long>(x1);
                auto tmp1 = static_cast<float>(tmp0);
                auto tmp2 = static_cast<float>(0.5);
                auto tmp3 = tmp1 + tmp2;
                auto tmp4 = static_cast<float>(1.2777777777777777);
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
                auto tmp25 = static_cast<float>(0.782608695652174);
                auto tmp26 = decltype(tmp24)(tmp24 * tmp25);
                auto tmp27 = std::abs(tmp26);
                auto tmp28 = static_cast<float>(1.0);
                auto tmp29 = min_propagate_nan(tmp27, tmp28);
                auto tmp30 = tmp28 - tmp29;
                auto tmp31 = static_cast<float>(0.0);
                auto tmp32 = tmp20 ? tmp30 : tmp31;
                auto tmp34 = tmp32 / tmp33;
                out_ptr4[static_cast<long>(x1 + (270L*x0))] = tmp34;
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(270L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(270L); x2+=static_cast<long>(1L))
                {
                    auto tmp64 = out_ptr4[static_cast<long>(x1)];
                    auto tmp66 = out_ptr4[static_cast<long>(270L + x1)];
                    auto tmp69 = out_ptr4[static_cast<long>(540L + x1)];
                    auto tmp72 = out_ptr4[static_cast<long>(810L + x1)];
                    auto tmp75 = out_ptr4[static_cast<long>(1080L + x1)];
                    auto tmp0 = static_cast<long>(x1);
                    auto tmp1 = static_cast<float>(tmp0);
                    auto tmp2 = static_cast<float>(0.5);
                    auto tmp3 = tmp1 + tmp2;
                    auto tmp4 = static_cast<float>(1.2777777777777777);
                    auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                    auto tmp6 = tmp5 - tmp4;
                    auto tmp7 = tmp6 + tmp2;
                    auto tmp8 = static_cast<long>(tmp7);
                    auto tmp9 = static_cast<long>(0);
                    auto tmp10 = max_propagate_nan(tmp8, tmp9);
                    auto tmp11 = tmp10 + tmp9;
                    auto tmp12 = static_cast<long>(344);
                    auto tmp13 = min_propagate_nan(tmp11, tmp12);
                    auto tmp14 = in_out_ptr0[static_cast<long>(x2 + (270L*tmp13) + (93150L*x0))];
                    auto tmp15 = static_cast<long>(x2);
                    auto tmp16 = static_cast<float>(tmp15);
                    auto tmp17 = tmp16 + tmp2;
                    auto tmp18 = static_cast<float>(1.6888888888888889);
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
                    auto tmp29 = static_cast<float>(tmp28);
                    auto tmp30 = static_cast<float>(0.0);
                    auto tmp31 = decltype(tmp29)(tmp29 * tmp30);
                    auto tmp32 = tmp14 + tmp31;
                    auto tmp33 = static_cast<long>(1);
                    auto tmp34 = tmp10 + tmp33;
                    auto tmp35 = min_propagate_nan(tmp34, tmp12);
                    auto tmp36 = in_out_ptr0[static_cast<long>(x2 + (270L*tmp35) + (93150L*x0))];
                    auto tmp37 = in_ptr0[static_cast<long>(x0 + (3L*tmp27) + (1368L*tmp35))];
                    auto tmp38 = static_cast<float>(tmp37);
                    auto tmp39 = decltype(tmp38)(tmp38 * tmp30);
                    auto tmp40 = tmp36 + tmp39;
                    auto tmp41 = static_cast<long>(2);
                    auto tmp42 = tmp10 + tmp41;
                    auto tmp43 = min_propagate_nan(tmp42, tmp12);
                    auto tmp44 = in_out_ptr0[static_cast<long>(x2 + (270L*tmp43) + (93150L*x0))];
                    auto tmp45 = in_ptr0[static_cast<long>(x0 + (3L*tmp27) + (1368L*tmp43))];
                    auto tmp46 = static_cast<float>(tmp45);
                    auto tmp47 = decltype(tmp46)(tmp46 * tmp30);
                    auto tmp48 = tmp44 + tmp47;
                    auto tmp49 = static_cast<long>(3);
                    auto tmp50 = tmp10 + tmp49;
                    auto tmp51 = min_propagate_nan(tmp50, tmp12);
                    auto tmp52 = in_out_ptr0[static_cast<long>(x2 + (270L*tmp51) + (93150L*x0))];
                    auto tmp53 = in_ptr0[static_cast<long>(x0 + (3L*tmp27) + (1368L*tmp51))];
                    auto tmp54 = static_cast<float>(tmp53);
                    auto tmp55 = decltype(tmp54)(tmp54 * tmp30);
                    auto tmp56 = tmp52 + tmp55;
                    auto tmp57 = tmp10 + tmp24;
                    auto tmp58 = min_propagate_nan(tmp57, tmp12);
                    auto tmp59 = in_out_ptr0[static_cast<long>(x2 + (270L*tmp58) + (93150L*x0))];
                    auto tmp60 = in_ptr0[static_cast<long>(x0 + (3L*tmp27) + (1368L*tmp58))];
                    auto tmp61 = static_cast<float>(tmp60);
                    auto tmp62 = decltype(tmp61)(tmp61 * tmp30);
                    auto tmp63 = tmp59 + tmp62;
                    auto tmp65 = decltype(tmp32)(tmp32 * tmp64);
                    auto tmp67 = decltype(tmp40)(tmp40 * tmp66);
                    auto tmp68 = tmp65 + tmp67;
                    auto tmp70 = decltype(tmp48)(tmp48 * tmp69);
                    auto tmp71 = tmp68 + tmp70;
                    auto tmp73 = decltype(tmp56)(tmp56 * tmp72);
                    auto tmp74 = tmp71 + tmp73;
                    auto tmp76 = decltype(tmp63)(tmp63 * tmp75);
                    auto tmp77 = tmp74 + tmp76;
                    in_out_ptr1[static_cast<long>(x2 + (270L*x1) + (72900L*x0))] = tmp77;
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(3L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(72900L); x1+=static_cast<long>(1L))
            {
                auto tmp0 = in_out_ptr1[static_cast<long>(x1 + (72900L*x0))];
                auto tmp1 = std::nearbyint(tmp0);
                auto tmp2 = static_cast<unsigned char>(tmp1);
                out_ptr10[static_cast<long>(x0 + (3L*x1))] = tmp2;
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
    buf1 = empty_strided((270, ), (1, ), device='cpu', dtype=torch.float32)
    buf3 = empty_strided((270, ), (1, ), device='cpu', dtype=torch.float32)
    buf4 = empty_strided((5, 270), (270, 1), device='cpu', dtype=torch.float32)
    buf5 = empty_strided((1, 3, 345, 270), (279450, 93150, 270, 1), device='cpu', dtype=torch.float32)
    buf6 = buf5; del buf5  # reuse
    buf8 = empty_strided((5, 270), (270, 1), device='cpu', dtype=torch.float32)
    buf10 = empty_strided((1, 3, 270, 270), (218700, 72900, 270, 1), device='cpu', dtype=torch.float32)
    buf13 = buf10; del buf10  # reuse
    buf14 = empty_strided((1, 3, 270, 270), (218700, 1, 810, 3), device='cpu', dtype=torch.uint8)
    cpp_fused__to_copy_add_clone_div_index_mul_round_sum_unbind_0(c_void_p(buf6.data_ptr()), c_void_p(buf13.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf1.data_ptr()), c_void_p(buf3.data_ptr()), c_void_p(buf4.data_ptr()), c_void_p(buf8.data_ptr()), c_void_p(buf14.data_ptr()))
    del arg0_1
    return (buf14, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 3, 345, 456), (471960, 1, 1368, 3), device='cpu', dtype=torch.uint8)
    return print_performance(lambda: call([arg0_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
