class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s0)", arg1_1: "Sym(s1)", arg2_1: "Sym(s2)", arg3_1: "f32[1, s0, s1, s2]", arg4_1: "Sym(s3)", arg5_1: "Sym(s4)"):
        # File: check_interpolate_bilinear.py:13, code: img = torch.nn.functional.interpolate(img, size=osize, mode="bilinear", antialias=False, align_corners=align_corners)
        sub: "Sym(s3)" = arg4_1 - 0
        add: "Sym(s3 + 1)" = sub + 1;  sub = None
        sub_1: "Sym(s3)" = add - 1;  add = None
        floordiv: "Sym(s3)" = sub_1 // 1;  sub_1 = None
        iota: "i64[s3]" = torch.ops.prims.iota.default(floordiv, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False);  floordiv = None
        sym_size_int_1: "Sym(s1)" = torch.ops.aten.sym_size.int(arg3_1, 2)
        sub_6: "Sym(s1 - 1.0)" = sym_size_int_1 - 1.0
        sub_7: "Sym(s3 - 1.0)" = arg4_1 - 1.0;  arg4_1 = None
        truediv_1: "Sym(s1/(s3 - 1.0) - 1.0/(s3 - 1.0))" = sub_6 / sub_7;  sub_6 = sub_7 = None
        mul_1: "f32[s3]" = torch.ops.aten.mul.Tensor(iota, truediv_1);  iota = truediv_1 = None
        clamp_min_1: "f32[s3]" = torch.ops.aten.clamp_min.default(mul_1, 0.0);  mul_1 = None
        unsqueeze: "f32[s3, 1]" = torch.ops.aten.unsqueeze.default(clamp_min_1, -1);  clamp_min_1 = None
        convert_element_type_1: "i64[s3, 1]" = torch.ops.prims.convert_element_type.default(unsqueeze, torch.int64)
        sub_9: "Sym(s1 - 1)" = sym_size_int_1 - 1;  sym_size_int_1 = None
        lt_1: "b8[s3, 1]" = torch.ops.aten.lt.Scalar(convert_element_type_1, sub_9);  sub_9 = None
        add_3: "i64[s3, 1]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1)
        where_1: "i64[s3, 1]" = torch.ops.aten.where.self(lt_1, add_3, convert_element_type_1);  lt_1 = add_3 = None
        sub_2: "Sym(s4)" = arg5_1 - 0
        add_1: "Sym(s4 + 1)" = sub_2 + 1;  sub_2 = None
        sub_3: "Sym(s4)" = add_1 - 1;  add_1 = None
        floordiv_1: "Sym(s4)" = sub_3 // 1;  sub_3 = None
        iota_1: "i64[s4]" = torch.ops.prims.iota.default(floordiv_1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False);  floordiv_1 = None
        sym_size_int: "Sym(s2)" = torch.ops.aten.sym_size.int(arg3_1, 3)
        sub_4: "Sym(s2 - 1.0)" = sym_size_int - 1.0
        sub_5: "Sym(s4 - 1.0)" = arg5_1 - 1.0;  arg5_1 = None
        truediv: "Sym(s2/(s4 - 1.0) - 1.0/(s4 - 1.0))" = sub_4 / sub_5;  sub_4 = sub_5 = None
        mul: "f32[s4]" = torch.ops.aten.mul.Tensor(iota_1, truediv);  iota_1 = truediv = None
        clamp_min: "f32[s4]" = torch.ops.aten.clamp_min.default(mul, 0.0);  mul = None
        convert_element_type: "i64[s4]" = torch.ops.prims.convert_element_type.default(clamp_min, torch.int64)
        sub_8: "Sym(s2 - 1)" = sym_size_int - 1;  sym_size_int = None
        lt: "b8[s4]" = torch.ops.aten.lt.Scalar(convert_element_type, sub_8);  sub_8 = None
        add_2: "i64[s4]" = torch.ops.aten.add.Tensor(convert_element_type, 1)
        where: "i64[s4]" = torch.ops.aten.where.self(lt, add_2, convert_element_type);  lt = add_2 = None
        _unsafe_index_3: "f32[1, s0, s3, s4]" = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, where_1, where])
        _unsafe_index_2: "f32[1, s0, s3, s4]" = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, where_1, convert_element_type]);  where_1 = None
        sub_13: "f32[1, s0, s3, s4]" = torch.ops.aten.sub.Tensor(_unsafe_index_3, _unsafe_index_2);  _unsafe_index_3 = None
        sub_11: "f32[s4]" = torch.ops.aten.sub.Tensor(clamp_min, convert_element_type);  clamp_min = None
        clamp_min_3: "f32[s4]" = torch.ops.aten.clamp_min.default(sub_11, 0.0);  sub_11 = None
        clamp_max_1: "f32[s4]" = torch.ops.aten.clamp_max.default(clamp_min_3, 1.0);  clamp_min_3 = None
        mul_3: "f32[1, s0, s3, s4]" = torch.ops.aten.mul.Tensor(sub_13, clamp_max_1);  sub_13 = None
        add_5: "f32[1, s0, s3, s4]" = torch.ops.aten.add.Tensor(_unsafe_index_2, mul_3);  _unsafe_index_2 = mul_3 = None
        _unsafe_index_1: "f32[1, s0, s3, s4]" = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, convert_element_type_1, where]);  where = None
        _unsafe_index: "f32[1, s0, s3, s4]" = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, convert_element_type_1, convert_element_type]);  arg3_1 = convert_element_type = None
        sub_12: "f32[1, s0, s3, s4]" = torch.ops.aten.sub.Tensor(_unsafe_index_1, _unsafe_index);  _unsafe_index_1 = None
        mul_2: "f32[1, s0, s3, s4]" = torch.ops.aten.mul.Tensor(sub_12, clamp_max_1);  sub_12 = clamp_max_1 = None
        add_4: "f32[1, s0, s3, s4]" = torch.ops.aten.add.Tensor(_unsafe_index, mul_2);  _unsafe_index = mul_2 = None
        sub_14: "f32[1, s0, s3, s4]" = torch.ops.aten.sub.Tensor(add_5, add_4);  add_5 = None
        sub_10: "f32[s3, 1]" = torch.ops.aten.sub.Tensor(unsqueeze, convert_element_type_1);  unsqueeze = convert_element_type_1 = None
        clamp_min_2: "f32[s3, 1]" = torch.ops.aten.clamp_min.default(sub_10, 0.0);  sub_10 = None
        clamp_max: "f32[s3, 1]" = torch.ops.aten.clamp_max.default(clamp_min_2, 1.0);  clamp_min_2 = None
        mul_4: "f32[1, s0, s3, s4]" = torch.ops.aten.mul.Tensor(sub_14, clamp_max);  sub_14 = clamp_max = None
        add_6: "f32[1, s0, s3, s4]" = torch.ops.aten.add.Tensor(add_4, mul_4);  add_4 = mul_4 = None
        return (add_6,)
        