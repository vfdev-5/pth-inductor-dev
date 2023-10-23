#include "/tmp/torchinductor_root/bf/cbf262yqjxhzxmw7lov36xiiezas3czyjs7cdvyrvlrje4xcl2kd.h"
extern "C" void kernel(const float* in_ptr0,
                       float* out_ptr0,
                       float* out_ptr1,
                       float* out_ptr2,
                       float* out_ptr3)
{
    // Horiz weights
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

    // Horiz pass
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(1380L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(272L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr0[static_cast<long>(5L*x2)];
                    auto tmp17 = out_ptr0[static_cast<long>(1L + (5L*x2))];
                    auto tmp24 = out_ptr0[static_cast<long>(2L + (5L*x2))];
                    auto tmp31 = out_ptr0[static_cast<long>(3L + (5L*x2))];
                    auto tmp38 = out_ptr0[static_cast<long>(4L + (5L*x2))];
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
                    out_ptr1[static_cast<long>(x2 + (272L*x1) + (816L*x0))] = tmp44;
                }
            }
        }
    }

    // Vertical pass
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(271L); x2+=static_cast<long>(1L))
                {
                    #pragma GCC ivdep
                    for(long x3=static_cast<long>(0L); x3<static_cast<long>(272L); x3+=static_cast<long>(1L))
                    {
                        auto tmp0 = c10::convert<long>(x2);
                        auto tmp1 = c10::convert<float>(tmp0);
                        auto tmp2 = c10::convert<float>(0.5);
                        auto tmp3 = tmp1 + tmp2;
                        auto tmp4 = c10::convert<float>(1.2730627306273063);
                        auto tmp5 = decltype(tmp3)(tmp3 * tmp4);
                        auto tmp6 = tmp5 + tmp4;
                        auto tmp7 = tmp6 + tmp2;
                        auto tmp8 = c10::convert<long>(tmp7);
                        auto tmp9 = c10::convert<long>(345);
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
                        auto tmp24 = c10::convert<float>(0.7855072463768116);
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
                        auto tmp80 = tmp31 / tmp79;
                        auto tmp81 = tmp15 + tmp14;
                        auto tmp82 = c10::convert<long>(344);
                        auto tmp83 = min_propagate_nan(tmp81, tmp82);
                        auto tmp84 = out_ptr1[static_cast<long>(x3 + (272L*x1) + (816L*tmp83) + (281520L*x0))];
                        auto tmp85 = decltype(tmp80)(tmp80 * tmp84);
                        auto tmp86 = tmp42 / tmp79;
                        auto tmp87 = tmp15 + tmp32;
                        auto tmp88 = min_propagate_nan(tmp87, tmp82);
                        auto tmp89 = out_ptr1[static_cast<long>(x3 + (272L*x1) + (816L*tmp88) + (281520L*x0))];
                        auto tmp90 = decltype(tmp86)(tmp86 * tmp89);
                        auto tmp91 = tmp85 + tmp90;
                        auto tmp92 = tmp54 / tmp79;
                        auto tmp93 = tmp15 + tmp44;
                        auto tmp94 = min_propagate_nan(tmp93, tmp82);
                        auto tmp95 = out_ptr1[static_cast<long>(x3 + (272L*x1) + (816L*tmp94) + (281520L*x0))];
                        auto tmp96 = decltype(tmp92)(tmp92 * tmp95);
                        auto tmp97 = tmp91 + tmp96;
                        auto tmp98 = tmp66 / tmp79;
                        auto tmp99 = tmp15 + tmp56;
                        auto tmp100 = min_propagate_nan(tmp99, tmp82);
                        auto tmp101 = out_ptr1[static_cast<long>(x3 + (272L*x1) + (816L*tmp100) + (281520L*x0))];
                        auto tmp102 = decltype(tmp98)(tmp98 * tmp101);
                        auto tmp103 = tmp97 + tmp102;
                        auto tmp104 = tmp78 / tmp79;
                        auto tmp105 = tmp15 + tmp68;
                        auto tmp106 = min_propagate_nan(tmp105, tmp82);
                        auto tmp107 = out_ptr1[static_cast<long>(x3 + (272L*x1) + (816L*tmp106) + (281520L*x0))];
                        auto tmp108 = decltype(tmp104)(tmp104 * tmp107);
                        auto tmp109 = tmp103 + tmp108;
                        out_ptr2[static_cast<long>(x3 + (272L*x2) + (73712L*x1) + (221136L*x0))] = tmp109;
                    }
                }
            }
        }
    }

    // Output to CL
    {
        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(4L); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(3L); x1+=static_cast<long>(1L))
            {
                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(73712L); x2+=static_cast<long>(1L))
                {
                    auto tmp0 = out_ptr2[static_cast<long>(x2 + (73712L*x1) + (221136L*x0))];
                    out_ptr3[static_cast<long>(x1 + (3L*x2) + (221136L*x0))] = tmp0;
                }
            }
        }
    }
}