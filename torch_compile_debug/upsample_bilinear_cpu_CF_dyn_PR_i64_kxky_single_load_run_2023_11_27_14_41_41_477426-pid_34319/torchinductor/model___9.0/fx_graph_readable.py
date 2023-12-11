class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s0)", arg1_1: "Sym(s1)", arg2_1: "Sym(s2)", arg3_1: "Sym(s3)", arg4_1: "u8[s0, s1, s2, s3]", arg5_1: "Sym(s4)", arg6_1: "Sym(s5)"):
        # File: check_interpolate_bilinear.py:12, code: img = torch.nn.functional.interpolate(img, size=osize, mode="bilinear", antialias=False, align_corners=False)
        sub: "Sym(s4)" = arg5_1 - 0
        add: "Sym(s4 + 1)" = sub + 1;  sub = None
        sub_1: "Sym(s4)" = add - 1;  add = None
        floordiv: "Sym(s4)" = sub_1 // 1;  sub_1 = None
        iota: "i64[s4]" = torch.ops.prims.iota.default(floordiv, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        sub_2: "Sym(s5)" = arg6_1 - 0
        add_1: "Sym(s5 + 1)" = sub_2 + 1;  sub_2 = None
        sub_3: "Sym(s5)" = add_1 - 1;  add_1 = None
        floordiv_1: "Sym(s5)" = sub_3 // 1;  sub_3 = None
        iota_1: "i64[s5]" = torch.ops.prims.iota.default(floordiv_1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False);  floordiv_1 = None
        add_2: "f32[s5]" = torch.ops.aten.add.Tensor(iota_1, 0.5);  iota_1 = None
        truediv: "Sym(s3/s5)" = arg3_1 / arg6_1
        mul: "f32[s5]" = torch.ops.aten.mul.Tensor(add_2, truediv);  add_2 = truediv = None
        sub_4: "f32[s5]" = torch.ops.aten.sub.Tensor(mul, 0.5);  mul = None
        clamp_min: "f32[s5]" = torch.ops.aten.clamp_min.default(sub_4, 0.0);  sub_4 = None
        add_3: "f32[s4]" = torch.ops.aten.add.Tensor(iota, 0.5);  iota = None
        truediv_1: "Sym(s2/s4)" = arg2_1 / arg5_1
        mul_1: "f32[s4]" = torch.ops.aten.mul.Tensor(add_3, truediv_1);  add_3 = truediv_1 = None
        sub_5: "f32[s4]" = torch.ops.aten.sub.Tensor(mul_1, 0.5);  mul_1 = None
        clamp_min_1: "f32[s4]" = torch.ops.aten.clamp_min.default(sub_5, 0.0);  sub_5 = None
        convert_element_type: "i64[s5]" = torch.ops.prims.convert_element_type.default(clamp_min, torch.int64)
        convert_element_type_1: "i64[s4]" = torch.ops.prims.convert_element_type.default(clamp_min_1, torch.int64)
        iota_2: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        iota_3: "i64[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        unsqueeze: "i64[s5, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1)
        add_4: "i64[s5, 2]" = torch.ops.aten.add.Tensor(unsqueeze, iota_2);  unsqueeze = iota_2 = None
        sub_6: "Sym(s3 - 1)" = arg3_1 - 1;  arg3_1 = None
        clamp_max: "i64[s5, 2]" = torch.ops.aten.clamp_max.default(add_4, sub_6);  add_4 = sub_6 = None
        unsqueeze_1: "i64[s4, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, -1)
        add_5: "i64[s4, 2]" = torch.ops.aten.add.Tensor(unsqueeze_1, iota_3);  unsqueeze_1 = iota_3 = None
        sub_7: "Sym(s2 - 1)" = arg2_1 - 1;  arg2_1 = None
        clamp_max_1: "i64[s4, 2]" = torch.ops.aten.clamp_max.default(add_5, sub_7);  add_5 = sub_7 = None
        sub_8: "f32[s5]" = torch.ops.aten.sub.Tensor(clamp_min, convert_element_type);  clamp_min = convert_element_type = None
        clamp_min_2: "f32[s5]" = torch.ops.aten.clamp_min.default(sub_8, 0.0);  sub_8 = None
        clamp_max_2: "f32[s5]" = torch.ops.aten.clamp_max.default(clamp_min_2, 1.0);  clamp_min_2 = None
        sub_9: "f32[s5]" = torch.ops.aten.sub.Tensor(1.0, clamp_max_2)
        unsqueeze_2: "f32[s5, 1]" = torch.ops.aten.unsqueeze.default(sub_9, 1);  sub_9 = None
        unsqueeze_3: "f32[s5, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_2, 1);  clamp_max_2 = None
        cat: "f32[s5, 2]" = torch.ops.aten.cat.default([unsqueeze_2, unsqueeze_3], -1);  unsqueeze_2 = unsqueeze_3 = None
        sub_10: "f32[s4]" = torch.ops.aten.sub.Tensor(clamp_min_1, convert_element_type_1);  clamp_min_1 = convert_element_type_1 = None
        clamp_min_3: "f32[s4]" = torch.ops.aten.clamp_min.default(sub_10, 0.0);  sub_10 = None
        clamp_max_3: "f32[s4]" = torch.ops.aten.clamp_max.default(clamp_min_3, 1.0);  clamp_min_3 = None
        sub_11: "f32[s4]" = torch.ops.aten.sub.Tensor(1.0, clamp_max_3)
        unsqueeze_4: "f32[s4, 1]" = torch.ops.aten.unsqueeze.default(sub_11, 1);  sub_11 = None
        unsqueeze_5: "f32[s4, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_3, 1);  clamp_max_3 = None
        cat_1: "f32[s4, 2]" = torch.ops.aten.cat.default([unsqueeze_4, unsqueeze_5], -1);  unsqueeze_4 = unsqueeze_5 = None
        view: "i64[s4, 2, 1, 1]" = torch.ops.aten.view.default(clamp_max_1, [floordiv, 2, 1, 1]);  clamp_max_1 = floordiv = None
        _unsafe_index: "u8[s0, s1, s4, 2, s5, 2]" = torch.ops.aten._unsafe_index.Tensor(arg4_1, [None, None, view, clamp_max]);  arg4_1 = view = clamp_max = None
        view_1: "f32[s4, 2, 1, 1]" = torch.ops.aten.view.default(cat_1, [arg5_1, 2, 1, 1]);  cat_1 = arg5_1 = None
        view_2: "f32[1, s5, 2]" = torch.ops.aten.view.default(cat, [1, arg6_1, 2]);  cat = arg6_1 = None
        mul_2: "f32[s0, s1, s4, 2, s5, 2]" = torch.ops.aten.mul.Tensor(view_2, _unsafe_index);  view_2 = _unsafe_index = None
        mul_3: "f32[s0, s1, s4, 2, s5, 2]" = torch.ops.aten.mul.Tensor(view_1, mul_2);  view_1 = mul_2 = None
        sum_1: "f32[s0, s1, s4, s5]" = torch.ops.aten.sum.dim_IntList(mul_3, [-1, -3]);  mul_3 = None
        round_1: "f32[s0, s1, s4, s5]" = torch.ops.aten.round.default(sum_1);  sum_1 = None
        convert_element_type_2: "u8[s0, s1, s4, s5]" = torch.ops.prims.convert_element_type.default(round_1, torch.uint8);  round_1 = None
        return (convert_element_type_2,)
        