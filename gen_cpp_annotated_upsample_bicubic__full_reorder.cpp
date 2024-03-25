#include <cmath>

#define c10_convert static_cast
#define max_propagate_nan std::max
#define min_propagate_nan std::min


void kernel(float* in_out_ptr1,
            const float* in_ptr0,
            const long ks0,
            const long ks1,
            const long ks2,
            const long ks3,
            const long ks4)
{
    {
        // Define constants
        constexpr auto tmp6 = static_cast<long>(1);
        constexpr auto tmp8 = static_cast<long>(0);

        constexpr auto tmp23 = static_cast<float>(0.0);
        constexpr auto tmp25 = static_cast<float>(1.0);
        constexpr auto tmp27 = static_cast<float>(1.25);
        constexpr auto tmp29 = static_cast<float>(2.25);
        constexpr auto tmp40 = static_cast<long>(2);
        constexpr auto tmp51 = static_cast<float>(-0.75);
        constexpr auto tmp53 = static_cast<float>(-3.75);
        constexpr auto tmp56 = static_cast<float>(-6.0);
        constexpr auto tmp59 = static_cast<float>(-3.0);
        constexpr auto tmp77 = static_cast<float>(2.0);

        auto tmp2 = c10_convert<float>(((-1.00000000000000)*(1.0/((-1.00000000000000) + ks1))) + (ks3*(1.0/((-1.00000000000000) + ks1))));
        auto tmp10 = c10_convert<long>((-1L) + ks3);
        auto tmp14 = c10_convert<float>(((-1.00000000000000)*(1.0/((-1.00000000000000) + ks2))) + (ks4*(1.0/((-1.00000000000000) + ks2))));
        auto tmp19 = c10_convert<long>((-1L) + ks4);

        #pragma GCC ivdep
        for(long x0=static_cast<long>(0L); x0<static_cast<long>(ks0); x0+=static_cast<long>(1L))
        {
            #pragma GCC ivdep
            for(long x1=static_cast<long>(0L); x1<static_cast<long>(ks1); x1+=static_cast<long>(1L))
            {

                // source_index_y_float = x1 * scale1
                auto tmp0 = c10_convert<long>(x1);
                auto tmp1 = c10_convert<float>(tmp0);
                auto tmp3 = decltype(tmp1)(tmp1 * tmp2);

                // source_index_y = (long) floor(source_index_y_float)
                auto tmp4 = std::floor(tmp3);
                auto tmp5 = c10_convert<long>(tmp4);

                // index_y_0 = clamp(source_index_y - 1, 0, ks3 - 1)
                auto tmp7 = decltype(tmp5)(tmp5 - tmp6);
                auto tmp9 = max_propagate_nan(tmp7, tmp8);
                auto tmp11 = min_propagate_nan(tmp9, tmp10);

                // index_y_1 = clamp(source_index_y, 0, ks3 - 1)
                auto tmp98 = max_propagate_nan(tmp5, tmp8);
                auto tmp99 = min_propagate_nan(tmp98, tmp10);

                // index_y_2 = clamp(source_index_y + 1, 0, ks3 - 1)
                auto tmp35 = decltype(tmp5)(tmp5 + tmp6);
                auto tmp36 = max_propagate_nan(tmp35, tmp8);
                auto tmp37 = min_propagate_nan(tmp36, tmp10);

                // index_y_3 = clamp(source_index_y + 2, 0, ks3 - 1)
                auto tmp41 = decltype(tmp5)(tmp5 + tmp40);
                auto tmp42 = max_propagate_nan(tmp41, tmp8);
                auto tmp43 = min_propagate_nan(tmp42, tmp10);

                // lambda_y = clamp(source_index_float_y - source_index_y, 0.0, 1.0)
                auto tmp111 = decltype(tmp3)(tmp3 - tmp4);
                auto tmp112 = max_propagate_nan(tmp111, tmp23);
                auto tmp113 = min_propagate_nan(tmp112, tmp25);

                // wy_0 = _upsample_cubic_convolution2(lambda_y + 1, A)
                auto tmp114 = decltype(tmp113)(tmp113 + tmp25);
                auto tmp115 = decltype(tmp114)(tmp114 * tmp51);
                auto tmp116 = decltype(tmp115)(tmp115 - tmp53);
                auto tmp117 = decltype(tmp116)(tmp116 * tmp114);
                auto tmp118 = decltype(tmp117)(tmp117 + tmp56);
                auto tmp119 = decltype(tmp118)(tmp118 * tmp114);
                auto tmp120 = decltype(tmp119)(tmp119 - tmp59);

                // wy_1 = _upsample_cubic_convolution1(lambda_y, A)
                auto tmp125 = decltype(tmp113)(tmp113 * tmp27);
                auto tmp126 = decltype(tmp125)(tmp125 - tmp29);
                auto tmp127 = decltype(tmp126)(tmp126 * tmp113);
                auto tmp128 = decltype(tmp127)(tmp127 * tmp113);
                auto tmp129 = decltype(tmp128)(tmp128 + tmp25);

                // wy_2 = _upsample_cubic_convolution1(1.0 - lambda_y, A)
                auto tmp135 = decltype(tmp25)(tmp25 - tmp113);
                auto tmp136 = decltype(tmp135)(tmp135 * tmp27);
                auto tmp137 = decltype(tmp136)(tmp136 - tmp29);
                auto tmp138 = decltype(tmp137)(tmp137 * tmp135);
                auto tmp139 = decltype(tmp138)(tmp138 * tmp135);
                auto tmp140 = decltype(tmp139)(tmp139 + tmp25);

                // wy_3 = _upsample_cubic_convolution2(2.0 - lambda_y, A)
                auto tmp146 = decltype(tmp77)(tmp77 - tmp113);
                auto tmp147 = decltype(tmp146)(tmp146 * tmp51);
                auto tmp148 = decltype(tmp147)(tmp147 - tmp53);
                auto tmp149 = decltype(tmp148)(tmp148 * tmp146);
                auto tmp150 = decltype(tmp149)(tmp149 + tmp56);
                auto tmp151 = decltype(tmp150)(tmp150 * tmp146);
                auto tmp152 = decltype(tmp151)(tmp151 - tmp59);

                #pragma GCC ivdep
                for(long x2=static_cast<long>(0L); x2<static_cast<long>(ks2); x2+=static_cast<long>(1L))
                {

                    // source_index_x_float = x2 * scale2
                    auto tmp12 = c10_convert<long>(x2);
                    auto tmp13 = c10_convert<float>(tmp12);
                    auto tmp15 = decltype(tmp13)(tmp13 * tmp14);
                    // source_index_x = (long) floor(source_index_float_x)
                    auto tmp16 = std::floor(tmp15);
                    auto tmp17 = c10_convert<long>(tmp16);

                    // ----
                    // index_x_0 = clamp(source_index_x - 1, 0, ks4 - 1)
                    auto tmp46 = decltype(tmp17)(tmp17 - tmp6);
                    auto tmp47 = max_propagate_nan(tmp46, tmp8);
                    auto tmp48 = min_propagate_nan(tmp47, tmp19);

                    // index_x_1 = clamp(source_index_x, 0, ks4 - 1)
                    auto tmp18 = max_propagate_nan(tmp17, tmp8);
                    auto tmp20 = min_propagate_nan(tmp18, tmp19);

                    // index_x_2 = clamp(source_index_x + 1, 0, ks4 - 1)
                    auto tmp62 = decltype(tmp17)(tmp17 + tmp6);
                    auto tmp63 = max_propagate_nan(tmp62, tmp8);
                    auto tmp64 = min_propagate_nan(tmp63, tmp19);

                    // index_x_3 = clamp(source_index_x + 2, 0, ks4 - 1)
                    auto tmp73 = decltype(tmp17)(tmp17 + tmp40);
                    auto tmp74 = max_propagate_nan(tmp73, tmp8);
                    auto tmp75 = min_propagate_nan(tmp74, tmp19);

                    // lambda_x = clamp(source_index_x_float - source_index_x, 0.0, 1.0)
                    auto tmp22 = decltype(tmp15)(tmp15 - tmp16);
                    auto tmp24 = max_propagate_nan(tmp22, tmp23);
                    auto tmp26 = min_propagate_nan(tmp24, tmp25);

                    // wx_0 = _upsample_cubic_convolution2(lambda_x + 1, A)
                    auto tmp50 = decltype(tmp26)(tmp26 + tmp25);
                    auto tmp52 = decltype(tmp50)(tmp50 * tmp51);
                    auto tmp54 = decltype(tmp52)(tmp52 - tmp53);
                    auto tmp55 = decltype(tmp54)(tmp54 * tmp50);
                    auto tmp57 = decltype(tmp55)(tmp55 + tmp56);
                    auto tmp58 = decltype(tmp57)(tmp57 * tmp50);
                    auto tmp60 = decltype(tmp58)(tmp58 - tmp59);

                    // wx_1 = _upsample_cubic_convolution1(lambda_x, A)
                    auto tmp28 = decltype(tmp26)(tmp26 * tmp27);
                    auto tmp30 = decltype(tmp28)(tmp28 - tmp29);
                    auto tmp31 = decltype(tmp30)(tmp30 * tmp26);
                    auto tmp32 = decltype(tmp31)(tmp31 * tmp26);
                    auto tmp33 = decltype(tmp32)(tmp32 + tmp25);

                    // wx_2 = _upsample_cubic_convolution1(1.0 - lambda_x, A)
                    auto tmp66 = decltype(tmp25)(tmp25 - tmp26);
                    auto tmp67 = decltype(tmp66)(tmp66 * tmp27);
                    auto tmp68 = decltype(tmp67)(tmp67 - tmp29);
                    auto tmp69 = decltype(tmp68)(tmp68 * tmp66);
                    auto tmp70 = decltype(tmp69)(tmp69 * tmp66);
                    auto tmp71 = decltype(tmp70)(tmp70 + tmp25);

                    // wx_3 = _upsample_cubic_convolution2(2.0 - lambda_x, A)
                    auto tmp78 = decltype(tmp77)(tmp77 - tmp26);
                    auto tmp79 = decltype(tmp78)(tmp78 * tmp51);
                    auto tmp80 = decltype(tmp79)(tmp79 - tmp53);
                    auto tmp81 = decltype(tmp80)(tmp80 * tmp78);
                    auto tmp82 = decltype(tmp81)(tmp81 + tmp56);
                    auto tmp83 = decltype(tmp82)(tmp82 * tmp78);
                    auto tmp84 = decltype(tmp83)(tmp83 - tmp59);


                    // v00 = input[index_y_0, index_x_0, x0]
                    auto tmp49 = in_ptr0[static_cast<long>(tmp48 + (ks4*tmp11) + (ks3*ks4*x0))];
                    // v01 = input[index_y_0, index_x_1, x0]
                    auto tmp21 = in_ptr0[static_cast<long>(tmp20 + (ks4*tmp11) + (ks3*ks4*x0))];
                    // v02 = input[index_y_0, index_x_2, x0]
                    auto tmp65 = in_ptr0[static_cast<long>(tmp64 + (ks4*tmp11) + (ks3*ks4*x0))];
                    // v03 = input[index_y_0, index_x_3, x0]
                    auto tmp76 = in_ptr0[static_cast<long>(tmp75 + (ks4*tmp11) + (ks3*ks4*x0))];

                    // v00 * wx_0
                    auto tmp61 = decltype(tmp49)(tmp49 * tmp60);
                    // v01 * wx_1
                    auto tmp34 = decltype(tmp21)(tmp21 * tmp33);
                    // v02 * wx_2
                    auto tmp72 = decltype(tmp65)(tmp65 * tmp71);
                    // v03 * wx_3
                    auto tmp85 = decltype(tmp76)(tmp76 * tmp84);



                    // v10 = input[index_y_1, index_x_0, x0]
                    auto tmp102 = in_ptr0[static_cast<long>(tmp48 + (ks4*tmp99) + (ks3*ks4*x0))];
                    // v11 = input[index_y_1, index_x_1, x0]
                    auto tmp100 = in_ptr0[static_cast<long>(tmp20 + (ks4*tmp99) + (ks3*ks4*x0))];
                    // v12 = input[index_y_1, index_x_2, x0]
                    auto tmp104 = in_ptr0[static_cast<long>(tmp64 + (ks4*tmp99) + (ks3*ks4*x0))];
                    // v13 = input[index_y_1, index_x_3, x0]
                    auto tmp106 = in_ptr0[static_cast<long>(tmp75 + (ks4*tmp99) + (ks3*ks4*x0))];

                    // v10 * wx_0
                    auto tmp103 = decltype(tmp102)(tmp102 * tmp60);
                    // v11 * wx_1
                    auto tmp101 = decltype(tmp100)(tmp100 * tmp33);
                    // v12 * wx_2
                    auto tmp105 = decltype(tmp104)(tmp104 * tmp71);
                    // v13 * wx_3
                    auto tmp107 = decltype(tmp106)(tmp106 * tmp84);



                    // v20 = input[index_y_2, index_x_0, x0]
                    auto tmp86 = in_ptr0[static_cast<long>(tmp48 + (ks4*tmp37) + (ks3*ks4*x0))];
                    // v21 = input[index_y_2, index_x_1, x0]
                    auto tmp38 = in_ptr0[static_cast<long>(tmp20 + (ks4*tmp37) + (ks3*ks4*x0))];
                    // v22 = input[index_y_2, index_x_2, x0]
                    auto tmp88 = in_ptr0[static_cast<long>(tmp64 + (ks4*tmp37) + (ks3*ks4*x0))];
                    // v23 = input[index_y_2, index_x_3, x0]
                    auto tmp90 = in_ptr0[static_cast<long>(tmp75 + (ks4*tmp37) + (ks3*ks4*x0))];

                    // v20 * wx_0
                    auto tmp87 = decltype(tmp86)(tmp86 * tmp60);
                    // v21 * wx_1
                    auto tmp39 = decltype(tmp38)(tmp38 * tmp33);
                    // v22 * wx_2
                    auto tmp89 = decltype(tmp88)(tmp88 * tmp71);
                    // v23 * wx_3
                    auto tmp91 = decltype(tmp90)(tmp90 * tmp84);



                    // v30 = input[index_y_3, index_x_0, x0]
                    auto tmp92 = in_ptr0[static_cast<long>(tmp48 + (ks4*tmp43) + (ks3*ks4*x0))];
                    // v31 = input[index_y_3, index_x_1, x0]
                    auto tmp44 = in_ptr0[static_cast<long>(tmp20 + (ks4*tmp43) + (ks3*ks4*x0))];
                    // v32 = input[index_y_3, index_x_2, x0]
                    auto tmp94 = in_ptr0[static_cast<long>(tmp64 + (ks4*tmp43) + (ks3*ks4*x0))];
                    // v33 = input[index_y_3, index_x_3, x0]
                    auto tmp96 = in_ptr0[static_cast<long>(tmp75 + (ks4*tmp43) + (ks3*ks4*x0))];

                    // v30 * wx_0
                    auto tmp93 = decltype(tmp92)(tmp92 * tmp60);
                    // v31 * wx_1
                    auto tmp45 = decltype(tmp44)(tmp44 * tmp33);
                    // v32 * wx_2
                    auto tmp95 = decltype(tmp94)(tmp94 * tmp71);
                    // v33 * wx_3
                    auto tmp97 = decltype(tmp96)(tmp96 * tmp84);


                    // ----
                    // out_0 =  v00 * wx_0 + v01 * wx_1 + v02 * wx_2 + v03 * wx_3
                    auto tmp108 = decltype(tmp61)(tmp61 + tmp34);
                    auto tmp109 = decltype(tmp108)(tmp108 + tmp72);
                    auto tmp110 = decltype(tmp109)(tmp109 + tmp85);
                    // out_1 = v10 * wx_0 + v11 * wx_1 + v12 * wx_2 + v13 * wx_3
                    auto tmp122 = decltype(tmp103)(tmp103 + tmp101);
                    auto tmp123 = decltype(tmp122)(tmp122 + tmp105);
                    auto tmp124 = decltype(tmp123)(tmp123 + tmp107);
                    // out_2 = v20 * wx_0 + v21 * wx_1 + v22 * wx_2 + v23 * wx_3
                    auto tmp132 = decltype(tmp87)(tmp87 + tmp39);
                    auto tmp133 = decltype(tmp132)(tmp132 + tmp89);
                    auto tmp134 = decltype(tmp133)(tmp133 + tmp91);
                    // out_3 = v30 * wx_0 + v31 * wx_1 + v32 * wx_2 + v33 * wx_3
                    auto tmp143 = decltype(tmp93)(tmp93 + tmp45);
                    auto tmp144 = decltype(tmp143)(tmp143 + tmp95);
                    auto tmp145 = decltype(tmp144)(tmp144 + tmp97);


                    // out_0 * wy_0
                    auto tmp121 = decltype(tmp110)(tmp110 * tmp120);
                    // out_1 * wy_1
                    auto tmp130 = decltype(tmp124)(tmp124 * tmp129);
                    // out_2 * wy_2
                    auto tmp141 = decltype(tmp134)(tmp134 * tmp140);
                    // out_3 * wy_3
                    auto tmp153 = decltype(tmp145)(tmp145 * tmp152);


                    // out_0 * wy_0 + out_1 * wy_1
                    auto tmp131 = decltype(tmp121)(tmp121 + tmp130);
                    // out_0 * wy_0 + out_1 * wy_1 + out_2 * wy_2
                    auto tmp142 = decltype(tmp131)(tmp131 + tmp141);
                    // out_0 * wy_0 + out_1 * wy_1 + out_2 * wy_2 + out_3 * wy_3
                    auto tmp154 = decltype(tmp142)(tmp142 + tmp153);

                    // ----
                    // output = out_0 * wy_0 + out_1 * wy_1 + out_2 * wy_2 + out_3 * wy_3
                    in_out_ptr1[static_cast<long>(x2 + (ks2*x1) + (ks1*ks2*x0))] = tmp154;
                }
            }
        }
    }
}