
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, as_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
async_compile = AsyncCompile()


cpp_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/mq/cmqzxwuyo7ryvun3egqos5jq5ak4fue7d2jbopbqs7pgpkhdpfh4.h"
#include <iostream>
#include <stdlib.h>
#include <iomanip>

extern "C" void kernel(float* in_out_ptr0,
                       const float* in_ptr0,
                       float* out_ptr1)
{
    auto out_ptr0 = in_out_ptr0;
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(3L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(224L); i1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i2=static_cast<long>(0L); i2<static_cast<long>(224L); i2+=static_cast<long>(1L))
                {
                    // i = torch.arange(out_h, dtype=input.dtype, device=input.device)
                    auto tmp0 = static_cast<long>(i1);
                    auto tmp1 = static_cast<double>(tmp0);

                    // ??? (1.0d * i + 0.0d)
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = tmp3 + tmp4;

                    // x = (h_scale_factor * (i + 0.5) - 0.5).clamp(min=0.0)
                    // -> (i + 0.5)
                    auto tmp6 = static_cast<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.5);
                    auto tmp8 = tmp6 + tmp7;
                    // -> h_scale_factor * (i + 0.5)
                    auto tmp9 = static_cast<float>(1.5401785714285714);  // 345 / 224
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    // -> h_scale_factor * (i + 0.5) - 0.5
                    auto tmp11 = tmp10 - tmp7;
                    // -> max(0, h_scale_factor * (i + 0.5) - 0.5)
                    auto tmp12 = static_cast<float>(0.0);
                    auto tmp13 = max_propagate_nan(tmp11, tmp12);

                    // x_floor = x.to(torch.int64)
                    auto tmp14 = static_cast<long>(tmp13);


                    // j = torch.arange(out_w, dtype=input.dtype, device=input.device)
                    auto tmp15 = static_cast<long>(i2);
                    auto tmp16 = static_cast<double>(tmp15);

                    // ??? (1.0d * j + 0.0d)
                    auto tmp17 = decltype(tmp16)(tmp16 * tmp2);
                    auto tmp18 = tmp17 + tmp4;

                    // y = (w_scale_factor * (j + 0.5) - 0.5).clamp(min=0.0)
                    // -> (j + 0.5)
                    auto tmp19 = static_cast<float>(tmp18);
                    auto tmp20 = tmp19 + tmp7;
                    // -> w_scale_factor * (j + 0.5)
                    auto tmp21 = static_cast<float>(2.0357142857142856);  // 456 / 224
                    auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                    // -> w_scale_factor * (j + 0.5) - 0.5
                    auto tmp23 = tmp22 - tmp7;
                    // max(0, w_scale_factor * (j + 0.5) - 0.5)
                    auto tmp24 = max_propagate_nan(tmp23, tmp12);

                    // y_floor = y.to(torch.int64)
                    auto tmp25 = static_cast<long>(tmp24);

                    // x_floor_view = x_floor.unsqueeze(1)
                    // v1 = aten._unsafe_index(input, [None, None, x_floor_view, y_floor])
                    auto tmp26 = in_ptr0[static_cast<long>(i0 + (3L*tmp25) + (1368L*tmp14))];

                    // xscale2 = x_view - x_floor_view
                    auto tmp27 = static_cast<float>(tmp14);
                    auto tmp28 = tmp13 - tmp27;
                    // xscale1 = 1.0 - xscale2
                    auto tmp29 = static_cast<float>(1.0);
                    auto tmp30 = tmp29 - tmp28;

                    // v1 * xscale1
                    auto tmp31 = decltype(tmp26)(tmp26 * tmp30);

                    // x_ceil = torch.ceil(x).clamp(max=in_h - 1).to(torch.int64)
                    auto tmp32 = std::ceil(tmp13);
                    auto tmp33 = static_cast<float>(344.0);
                    auto tmp34 = min_propagate_nan(tmp32, tmp33);
                    auto tmp35 = static_cast<long>(tmp34);

                    // x_ceil_view = x_ceil.unsqueeze(1)
                    // v2 = aten._unsafe_index(input, [None, None, x_ceil_view, y_floor])
                    auto tmp36 = in_ptr0[static_cast<long>(i0 + (3L*tmp25) + (1368L*tmp35))];

                    // v2 * xscale2
                    auto tmp37 = decltype(tmp36)(tmp36 * tmp28);

                    // q1 = torch.mul(v1, xscale1) + torch.mul(v2, xscale2)
                    auto tmp38 = tmp31 + tmp37;

                    //
                    out_ptr0[static_cast<long>(i2 + (224L*i1) + (50176L*i0))] = tmp38;

                    /*
                    // if (i0 == 0 && (8 < i1 && i1 < 10) && (8 < i2 && i2 < 11)) {
                    // if (i0 == 0 && (-1 < i1 && i1 < 1) && (9 < i2 && i2 < 11)) {
                    if (i0 == 0 && (i1 == 0) && (i2 == 31)) {
                        auto v1 = tmp26;
                        auto v2 = tmp36;
                        // auto v3 = ttmp29;
                        // auto v4 = ttmp39;

                        auto xscale1 = tmp30;
                        auto xscale2 = tmp28;

                        // auto yscale1 = tttmp19;
                        // auto yscale2 = tttmp17;

                        auto q1 = tmp38;
                        // auto q2 = ttmp41;
                        // auto result = tttmp23;

                        std::cout << "3-blocks (a):" << std::fixed << std::setprecision(8)
                                  << std::endl << "v1: " << v1
                                  << std::endl << "v2: " << v2
                                  // << std::endl << "v3: " << v3
                                  // << std::endl << "v4: " << v4
                                  << std::endl << "xscale1: " << (double) xscale1
                                  << std::endl << "xscale2: " << (double) xscale2
                                  // << std::endl << "yscale1: " << yscale1
                                  // << std::endl << "yscale2: " << yscale2
                                  << std::endl << "q1: " << (double) q1
                                  // << std::endl << "q2: " << q2
                                  // << std::endl << "result: " << result
                                  << std::endl;
                    }
                    */
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(3L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(224L); i1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i2=static_cast<long>(0L); i2<static_cast<long>(224L); i2+=static_cast<long>(1L))
                {
                    // i = torch.arange(out_h, dtype=input.dtype, device=input.device)
                    auto tmp0 = static_cast<long>(i1);
                    auto tmp1 = static_cast<double>(tmp0);
                    // ???
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = tmp3 + tmp4;

                    // x = (h_scale_factor * (i + 0.5) - 0.5).clamp(min=0.0)
                    auto tmp6 = static_cast<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.5);
                    auto tmp8 = tmp6 + tmp7;
                    auto tmp9 = static_cast<float>(1.5401785714285714);
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    auto tmp11 = tmp10 - tmp7;
                    auto tmp12 = static_cast<float>(0.0);
                    auto tmp13 = max_propagate_nan(tmp11, tmp12);

                    // x_floor = x.to(torch.int64)
                    auto tmp14 = static_cast<long>(tmp13);

                    // j = torch.arange(out_w, dtype=input.dtype, device=input.device)
                    auto tmp15 = static_cast<long>(i2);
                    auto tmp16 = static_cast<double>(tmp15);
                    auto tmp17 = decltype(tmp16)(tmp16 * tmp2);
                    auto tmp18 = tmp17 + tmp4;

                    // y = (w_scale_factor * (j + 0.5) - 0.5).clamp(min=0.0)
                    auto tmp19 = static_cast<float>(tmp18);
                    auto tmp20 = tmp19 + tmp7;
                    auto tmp21 = static_cast<float>(2.0357142857142856);
                    auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                    auto tmp23 = tmp22 - tmp7;
                    auto tmp24 = max_propagate_nan(tmp23, tmp12);

                    // y_ceil = torch.ceil(y).clamp(max=in_w - 1).to(torch.int64)
                    auto tmp25 = std::ceil(tmp24);
                    auto tmp26 = static_cast<float>(455.0);
                    auto tmp27 = min_propagate_nan(tmp25, tmp26);
                    auto tmp28 = static_cast<long>(tmp27);

                    // x_floor_view = x_floor.unsqueeze(1)
                    // v3 = aten._unsafe_index(input, [None, None, x_floor_view, y_ceil])
                    auto tmp29 = in_ptr0[static_cast<long>(i0 + (3L*tmp28) + (1368L*tmp14))];

                    // xscale2 = x_view - x_floor_view
                    auto tmp30 = static_cast<float>(tmp14);
                    auto tmp31 = tmp13 - tmp30;

                    // xscale1 = 1.0 - xscale2
                    auto tmp32 = static_cast<float>(1.0);
                    auto tmp33 = tmp32 - tmp31;

                    // torch.mul(v3, xscale1)
                    auto tmp34 = decltype(tmp29)(tmp29 * tmp33);

                    // x_ceil = torch.ceil(x).clamp(max=in_h - 1).to(torch.int64)
                    auto tmp35 = std::ceil(tmp13);
                    auto tmp36 = static_cast<float>(344.0);
                    auto tmp37 = min_propagate_nan(tmp35, tmp36);
                    auto tmp38 = static_cast<long>(tmp37);

                    // x_ceil_view = x_ceil.unsqueeze(1)
                    // v4 = aten._unsafe_index(input, [None, None, x_ceil_view, y_ceil])
                    auto tmp39 = in_ptr0[static_cast<long>(i0 + (3L*tmp28) + (1368L*tmp38))];

                    // torch.mul(v4, xscale2)
                    auto tmp40 = decltype(tmp39)(tmp39 * tmp31);

                    // q2 = torch.mul(v3, xscale1) + torch.mul(v4, xscale2)
                    auto tmp41 = tmp34 + tmp40;

                    //
                    out_ptr1[static_cast<long>(i2 + (224L*i1) + (50176L*i0))] = tmp41;

                    /*
                    // if (i0 == 0 && (8 < i1 && i1 < 10) && (8 < i2 && i2 < 11)) {
                    // if (i0 == 0 && (-1 < i1 && i1 < 1) && (9 < i2 && i2 < 11)) {
                    if (i0 == 0 && (i1 == 0) && (i2 == 31)) {
                        // auto v1 = tmp26;
                        // auto v2 = tmp36;
                        auto v3 = tmp29;
                        auto v4 = tmp39;

                        // auto xscale1 = tmp30;
                        // auto xscale2 = tmp28;

                        // auto yscale1 = tttmp19;
                        // auto yscale2 = tttmp17;

                        // auto q1 = tmp38;
                        auto q2 = tmp41;
                        // auto result = tttmp23;

                        std::cout << "3-blocks (b):" << std::fixed << std::setprecision(8)
                                  // << std::endl << "v1: " << v1
                                  // << std::endl << "v2: " << v2
                                  << std::endl << "v3: " << v3
                                  << std::endl << "v4: " << v4
                                  // << std::endl << "xscale1: " << xscale1
                                  // << std::endl << "xscale2: " << xscale2
                                  // << std::endl << "yscale1: " << yscale1
                                  // << std::endl << "yscale2: " << yscale2
                                  // << std::endl << "q1: " << q1
                                  << std::endl << "q2: " << (double) q2
                                  // << std::endl << "result: " << (double) result
                                  << std::endl;
                    }
                    */
                }
            }
        }
    }
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(672L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(224L); i1+=static_cast<long>(1L))
            {
                auto tmp0 = out_ptr0[static_cast<long>(i1 + (224L*i0))];
                auto tmp21 = out_ptr1[static_cast<long>(i1 + (224L*i0))];

                // j = torch.arange(out_w, dtype=input.dtype, device=input.device)
                auto tmp1 = static_cast<long>(i1);
                auto tmp2 = static_cast<double>(tmp1);
                auto tmp3 = static_cast<double>(1.0);
                auto tmp4 = decltype(tmp2)(tmp2 * tmp3);
                auto tmp5 = static_cast<double>(0.0);
                auto tmp6 = tmp4 + tmp5;

                // y = (w_scale_factor * (j + 0.5) - 0.5).clamp(min=0.0)
                auto tmp7 = static_cast<float>(tmp6);
                auto tmp8 = static_cast<float>(0.5);
                auto tmp9 = tmp7 + tmp8;
                auto tmp10 = static_cast<float>(2.0357142857142856);
                auto tmp11 = decltype(tmp9)(tmp9 * tmp10);
                auto tmp12 = tmp11 - tmp8;
                auto tmp13 = static_cast<float>(0.0);
                auto tmp14 = max_propagate_nan(tmp12, tmp13);

                /*
                if (i0 == 0 && i1 == 31) {
                    std::cout << std::fixed << std::setprecision(8)
                              << tmp6 << " "
                              << tmp7 << " "
                              << tmp8 << " "
                              << tmp9 << " "
                              << tmp10 << " "
                              << tmp11 << " "
                              << tmp12 << " "
                              << tmp13 << " "
                              << tmp14 << " " << std::endl;
                }
                */

                // y_floor = y.to(torch.int64)
                auto tmp15 = static_cast<long>(tmp14);

                // yscale2 = y - y_floor
                auto tmp16 = static_cast<float>(tmp15);
                auto tmp17 = tmp14 - tmp16;

                // yscale1 = 1.0 - yscale2
                auto tmp18 = static_cast<float>(1.0);
                auto tmp19 = tmp18 - tmp17;

                // result = torch.mul(q1, yscale1) + torch.mul(q2, yscale2)
                auto tmp20 = decltype(tmp0)(tmp0 * tmp19);
                auto tmp22 = decltype(tmp21)(tmp21 * tmp17);
                auto tmp23 = tmp20 + tmp22;

                //
                in_out_ptr0[static_cast<long>(i1 + (224L*i0))] = tmp23;

                // if ((8 < i0 && i0 < 10) && (8 < i1 && i1 < 11)) {
                // if (i0 == 0 && (9 < i1 && i1 < 11)) {
                if (i0 == 0 && i1 == 31) {
                    // auto v1 = tmp26;
                    // auto v2 = tmp36;
                    // auto v3 = tmp29;
                    // auto v4 = tmp39;

                    // auto xscale1 = tmp30;
                    // auto xscale2 = tmp28;

                    auto yscale1 = tmp19;
                    auto yscale2 = tmp17;

                    // auto q1 = tmp38;
                    // auto q2 = tmp41;
                    auto result = tmp23;

                    auto j = tmp6;
                    auto y = tmp14;
                    auto y_floor = tmp15;

                    std::cout << "3-blocks (c):" << std::fixed << std::setprecision(8)
                                // << std::endl << "v1: " << v1
                                // << std::endl << "v2: " << v2
                                // << std::endl << "v3: " << v3
                                // << std::endl << "v4: " << v4
                                // << std::endl << "xscale1: " << xscale1
                                // << std::endl << "xscale2: " << xscale2
                                << std::endl << "yscale1: " << (double) yscale1
                                << std::endl << "yscale2: " << (double) yscale2
                                // << std::endl << "q1: " << q1
                                // << std::endl << "q2: " << q2
                                << std::endl << "result: " << (double) result
                                << std::endl << "j: " << (double) j
                                << std::endl << "y: " << (double) y
                                << std::endl << "y_floor: " << (double) y_floor
                                << std::endl;
                }
            }
        }
    }
}
''')



v1_cpp_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_0 = async_compile.cpp('''
#include "/tmp/torchinductor_root/mq/cmqzxwuyo7ryvun3egqos5jq5ak4fue7d2jbopbqs7pgpkhdpfh4.h"
#include <iostream>
#include <stdlib.h>
#include <iomanip>

extern "C" void kernel(float* out_ptr0, const float* in_ptr0)
{
    {
        #pragma GCC ivdep
        for(long i0=static_cast<long>(0L); i0<static_cast<long>(3L); i0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long i1=static_cast<long>(0L); i1<static_cast<long>(224L); i1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long i2=static_cast<long>(0L); i2<static_cast<long>(224L); i2+=static_cast<long>(1L))
                {
                    // i = torch.arange(out_h, dtype=input.dtype, device=input.device)
                    auto tmp0 = static_cast<long>(i1);
                    auto tmp1 = static_cast<double>(tmp0);

                    // ??? (1.0d * i + 0.0d)
                    auto tmp2 = static_cast<double>(1.0);
                    auto tmp3 = decltype(tmp1)(tmp1 * tmp2);
                    auto tmp4 = static_cast<double>(0.0);
                    auto tmp5 = tmp3 + tmp4;

                    // x = (h_scale_factor * (i + 0.5) - 0.5).clamp(min=0.0)
                    // -> (i + 0.5)
                    auto tmp6 = static_cast<float>(tmp5);
                    auto tmp7 = static_cast<float>(0.5);
                    auto tmp8 = tmp6 + tmp7;
                    // -> h_scale_factor * (i + 0.5)
                    auto tmp9 = static_cast<float>(1.5401785714285714);  // 345 / 224
                    auto tmp10 = decltype(tmp8)(tmp8 * tmp9);
                    // -> h_scale_factor * (i + 0.5) - 0.5
                    auto tmp11 = tmp10 - tmp7;
                    // -> max(0, h_scale_factor * (i + 0.5) - 0.5)
                    auto tmp12 = static_cast<float>(0.0);
                    auto tmp13 = max_propagate_nan(tmp11, tmp12);

                    // x_floor = x.to(torch.int64)
                    auto tmp14 = static_cast<long>(tmp13);

                    // j = torch.arange(out_w, dtype=input.dtype, device=input.device)
                    auto tmp15 = static_cast<long>(i2);
                    auto tmp16 = static_cast<double>(tmp15);

                    // ??? (1.0d * j + 0.0d)
                    auto tmp17 = decltype(tmp16)(tmp16 * tmp2);
                    auto tmp18 = tmp17 + tmp4;

                    // y = (w_scale_factor * (j + 0.5) - 0.5).clamp(min=0.0)
                    // -> (j + 0.5)
                    auto tmp19 = static_cast<float>(tmp18);
                    auto tmp20 = tmp19 + tmp7;
                    // -> w_scale_factor * (j + 0.5)
                    auto tmp21 = static_cast<float>(2.0357142857142856);  // 456 / 224
                    auto tmp22 = decltype(tmp20)(tmp20 * tmp21);
                    // -> w_scale_factor * (j + 0.5) - 0.5
                    auto tmp23 = tmp22 - tmp7;
                    // max(0, w_scale_factor * (j + 0.5) - 0.5)
                    auto tmp24 = max_propagate_nan(tmp23, tmp12);

                    // y_floor = y.to(torch.int64)
                    auto tmp25 = static_cast<long>(tmp24);

                    // x_floor_view = x_floor.unsqueeze(1)
                    // v1 = aten._unsafe_index(input, [None, None, x_floor_view, y_floor])
                    auto tmp26 = in_ptr0[static_cast<long>(i0 + (3L*tmp25) + (1368L*tmp14))];

                    // xscale2 = x_view - x_floor_view
                    auto tmp27 = static_cast<float>(tmp14);
                    auto tmp28 = tmp13 - tmp27;

                    // xscale1 = 1.0 - xscale2
                    auto tmp29 = static_cast<float>(1.0);
                    auto tmp30 = tmp29 - tmp28;

                    // v1 * xscale1
                    auto tmp31 = decltype(tmp26)(tmp26 * tmp30);

                    // x_ceil = torch.ceil(x).clamp(max=in_h - 1).to(torch.int64)
                    auto tmp32 = std::ceil(tmp13);
                    auto tmp33 = static_cast<float>(344.0);
                    auto tmp34 = min_propagate_nan(tmp32, tmp33);
                    auto tmp35 = static_cast<long>(tmp34);

                    // x_ceil_view = x_ceil.unsqueeze(1)
                    // v2 = aten._unsafe_index(input, [None, None, x_ceil_view, y_floor])
                    auto tmp36 = in_ptr0[static_cast<long>(i0 + (3L*tmp25) + (1368L*tmp35))];

                    // v2 * xscale2
                    auto tmp37 = decltype(tmp36)(tmp36 * tmp28);

                    // q1 = torch.mul(v1, xscale1) + torch.mul(v2, xscale2)
                    auto tmp38 = tmp31 + tmp37;

                    // ---- Loop 2

                    // y_ceil = torch.ceil(y).clamp(max=in_w - 1).to(torch.int64)
                    auto ttmp25 = std::ceil(tmp24);
                    auto ttmp26 = static_cast<float>(455.0);
                    auto ttmp27 = min_propagate_nan(ttmp25, ttmp26);
                    auto ttmp28 = static_cast<long>(ttmp27);

                    // x_floor_view = x_floor.unsqueeze(1)
                    // v3 = aten._unsafe_index(input, [None, None, x_floor_view, y_ceil])
                    auto ttmp29 = in_ptr0[static_cast<long>(i0 + (3L*ttmp28) + (1368L*tmp14))];

                    // x_ceil_view = x_ceil.unsqueeze(1)
                    // v4 = aten._unsafe_index(input, [None, None, x_ceil_view, y_ceil])
                    auto ttmp39 = in_ptr0[static_cast<long>(i0 + (3L*ttmp28) + (1368L*tmp35))];

                    // torch.mul(v3, xscale1)
                    auto ttmp34 = decltype(ttmp29)(ttmp29 * tmp30);

                    // torch.mul(v4, xscale2)
                    auto ttmp40 = decltype(ttmp39)(ttmp39 * tmp28);

                    // q2 = torch.mul(v3, xscale1) + torch.mul(v4, xscale2)
                    auto ttmp41 = ttmp34 + ttmp40;

                    // ---- Loop 3

                    // yscale2 = y - y_floor
                    auto tttmp16 = static_cast<float>(tmp25);
                    auto tttmp17 = tmp24 - tttmp16;

                    // yscale1 = 1.0 - yscale2
                    auto tttmp18 = static_cast<float>(1.0);
                    auto tttmp19 = tttmp18 - tttmp17;

                    // result = torch.mul(q1, yscale1) + torch.mul(q2, yscale2)
                    auto tttmp20 = decltype(tmp38)(tmp38 * tttmp19);
                    auto tttmp22 = decltype(ttmp41)(ttmp41 * tttmp17);
                    auto tttmp23 = tttmp20 + tttmp22;

                    // if (i0 == 0 && (-1 < i1 && i1 < 1) && (9 < i2 && i2 < 11)) {
                    if (i0 == 0 && (i1 == 0) && (i2 == 31)) {
                        auto v1 = tmp26;
                        auto v2 = tmp36;
                        auto v3 = ttmp29;
                        auto v4 = ttmp39;

                        auto xscale1 = tmp30;
                        auto xscale2 = tmp28;
                        auto yscale1 = tttmp19;
                        auto yscale2 = tttmp17;

                        auto q1 = tmp38;
                        auto q2 = ttmp41;
                        auto result = tttmp23;

                        auto j = tmp18;
                        auto y = tmp24;
                        auto y_floor = tmp25;

                        std::cout << "1-block:" << std::fixed << std::setprecision(8)
                                  << std::endl << "v1: " << v1
                                  << std::endl << "v2: " << v2
                                  << std::endl << "v3: " << v3
                                  << std::endl << "v4: " << v4
                                  << std::endl << "xscale1: " << (double) xscale1
                                  << std::endl << "xscale2: " << (double) xscale2
                                  << std::endl << "yscale1: " << (double) yscale1
                                  << std::endl << "yscale2: " << (double) yscale2
                                  << std::endl << "q1: " << (double) q1
                                  << std::endl << "q2: " << (double) q2
                                  << std::endl << "result: " << (double) result
                                  << std::endl << "j: " << (double) j
                                  << std::endl << "y: " << (double) y
                                  << std::endl << "y_floor: " << (double) y_floor
                                  << std::endl;
                    }

                    //
                    out_ptr0[static_cast<long>(i2 + (224L*i1) + (50176L*i0))] = tttmp23;

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
    assert_size_stride(arg0_1, (3, 345, 456), (1, 1368, 3))
    buf0 = empty_strided((1, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    buf1 = empty_strided((1, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    buf2 = buf0; del buf0  # reuse
    cpp_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_0(c_void_p(buf2.data_ptr()), c_void_p(arg0_1.data_ptr()), c_void_p(buf1.data_ptr()))
    del arg0_1
    return (buf2, )


def call_v1(args):
    arg0_1, = args
    args.clear()
    assert_size_stride(arg0_1, (3, 345, 456), (1, 1368, 3))
    buf2 = empty_strided((1, 3, 224, 224), (150528, 50176, 224, 1), device='cpu', dtype=torch.float32)
    v1_cpp_fused__to_copy__unsafe_index_add_arange_clamp_mul_rsub_sub_0(c_void_p(buf2.data_ptr()), c_void_p(arg0_1.data_ptr()))
    del arg0_1
    return (buf2, )


def benchmark_compiled_module(times=1000, repeat=1000):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((3, 345, 456), (1, 1368, 3), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1]), times=times, repeat=repeat)


def benchmark_compiled_module_v1(times=1000, repeat=1000):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((3, 345, 456), (1, 1368, 3), device='cpu', dtype=torch.float32)
    return print_performance(lambda: call_v1([arg0_1]), times=times, repeat=repeat)


def transform(img):
    img = img[None, ...]
    img = torch.nn.functional.interpolate(img, size=(224, 224), mode="bilinear", antialias=False)
    return img


def func():

    # torch.manual_seed(12)
    # x = torch.randint(0, 256, size=(1, 3, 345, 456), dtype=torch.float32)

    torch.manual_seed(12)
    # x = torch.randint(-1000, 1000, size=(1, 3, 345, 456), dtype=torch.float32)
    x = torch.rand(1, 3, 345, 456)
    x = x.contiguous(memory_format=torch.channels_last)[0]

    output = call([x])[0]
    expected = transform(x)

    torch.testing.assert_close(output, expected)


def func_v1():

    torch.manual_seed(12)

    # x = torch.randint(0, 256, size=(1, 3, 345, 456), dtype=torch.float32)
    # x = torch.randint(-1000, 1000, size=(1, 3, 345, 456), dtype=torch.float32)
    x = torch.rand(1, 3, 345, 456)
    x = x.contiguous(memory_format=torch.channels_last)[0]

    output0 = call([x])[0]
    output = call_v1([x])[0]
    expected = transform(x)

    abs_diff = (output - output0).abs()
    print("abs_diff.max():", abs_diff.max())
    m = abs_diff > 1e-4
    nonmatching_indices = torch.where(m)
    print(nonmatching_indices[2][:10], nonmatching_indices[3][:10])
    print(output[m][:10])
    print(output0[m][:10])

    torch.testing.assert_close(output0, expected)
    torch.testing.assert_close(output, output0)
    torch.testing.assert_close(output, expected)
    # torch.testing.assert_close(output, expected, atol=1e-3, rtol=1e-6)

if __name__ == "__main__":

    print("Test v0")
    func()
    print("Test v1")
    func_v1()

    # print("Baseline:")
    # from torch._inductor.utils import compiled_module_main
    # compiled_module_main('None', benchmark_compiled_module)

    # print("v1:")
    # from torch._inductor.utils import compiled_module_main
    # compiled_module_main('None', benchmark_compiled_module_v1)




# >>> tmp6 = 31
# >>> tmp10 = 2.0357142857142856
# >>> tmp11 = (0.5 + tmp6) * tmp10
# >>> tmp11
# 64.125
# >>> print(tmp11)
# 64.125
# >>> "{:.6f}".format(tmp11)
# '64.125000'
# >>> "{:.8f}".format(tmp11)
# '64.12500000'
# >>> tmp12 = tmp11 - 0.5
# >>> "{:.8f}".format(tmp12)
# '63.62500000'