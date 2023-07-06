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

                    //
                    out_ptr0[static_cast<long>(i2 + (224L*i1) + (50176L*i0))] = tttmp23;

                }
            }
        }
    }
}