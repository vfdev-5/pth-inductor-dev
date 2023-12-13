class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s0)", arg1_1: "Sym(s1)", arg2_1: "Sym(s2)", arg3_1: "f32[1, s0, s1, s2]", arg4_1: "Sym(s3)", arg5_1: "Sym(s4)"):
        # File: check_interpolate_bilinear.py:13, code: img = torch.nn.functional.interpolate(img, size=osize, mode="bilinear", antialias=False, align_corners=align_corners)
        sub: "Sym(s3)" = arg4_1 - 0
        truediv: "Sym(s3)" = sub / 1;  sub = None
        ceil: "Sym(s3)" = math_ceil(truediv);  truediv = None
        iota: "i64[s3]" = torch.ops.prims.iota.default(ceil, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False);  ceil = None
        convert_element_type: "f64[s3]" = torch.ops.prims.convert_element_type.default(iota, torch.float64);  iota = None
        mul: "f64[s3]" = torch.ops.aten.mul.Tensor(convert_element_type, 1);  convert_element_type = None
        add: "f64[s3]" = torch.ops.aten.add.Tensor(mul, 0);  mul = None
        convert_element_type_1: "f32[s3]" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        sub_1: "Sym(s4)" = arg5_1 - 0
        truediv_1: "Sym(s4)" = sub_1 / 1;  sub_1 = None
        ceil_1: "Sym(s4)" = math_ceil(truediv_1);  truediv_1 = None
        iota_1: "i64[s4]" = torch.ops.prims.iota.default(ceil_1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False);  ceil_1 = None
        convert_element_type_2: "f64[s4]" = torch.ops.prims.convert_element_type.default(iota_1, torch.float64);  iota_1 = None
        mul_1: "f64[s4]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None
        add_1: "f64[s4]" = torch.ops.aten.add.Tensor(mul_1, 0);  mul_1 = None
        convert_element_type_3: "f32[s4]" = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        sym_size_int: "Sym(s1)" = torch.ops.aten.sym_size.int(arg3_1, 2)
        sub_2: "Sym(s1 - 1)" = sym_size_int - 1
        sub_3: "Sym(s3 - 1)" = arg4_1 - 1;  arg4_1 = None
        truediv_2: "Sym(s1/(s3 - 1) - 1/(s3 - 1))" = sub_2 / sub_3;  sub_2 = sub_3 = None
        mul_2: "f32[s3]" = torch.ops.aten.mul.Tensor(convert_element_type_1, truediv_2);  convert_element_type_1 = truediv_2 = None
        sym_size_int_1: "Sym(s2)" = torch.ops.aten.sym_size.int(arg3_1, 3)
        sub_4: "Sym(s2 - 1)" = sym_size_int_1 - 1
        sub_5: "Sym(s4 - 1)" = arg5_1 - 1;  arg5_1 = None
        truediv_3: "Sym(s2/(s4 - 1) - 1/(s4 - 1))" = sub_4 / sub_5;  sub_4 = sub_5 = None
        mul_3: "f32[s4]" = torch.ops.aten.mul.Tensor(convert_element_type_3, truediv_3);  convert_element_type_3 = truediv_3 = None
        convert_element_type_4: "i64[s3]" = torch.ops.prims.convert_element_type.default(mul_2, torch.int64)
        ceil_2: "f32[s3]" = torch.ops.aten.ceil.default(mul_2)
        sub_6: "Sym(s1 - 1)" = sym_size_int - 1;  sym_size_int = None
        clamp_max: "f32[s3]" = torch.ops.aten.clamp_max.default(ceil_2, sub_6);  ceil_2 = sub_6 = None
        convert_element_type_5: "i64[s3]" = torch.ops.prims.convert_element_type.default(clamp_max, torch.int64);  clamp_max = None
        convert_element_type_6: "i64[s4]" = torch.ops.prims.convert_element_type.default(mul_3, torch.int64)
        ceil_3: "f32[s4]" = torch.ops.aten.ceil.default(mul_3)
        sub_7: "Sym(s2 - 1)" = sym_size_int_1 - 1;  sym_size_int_1 = None
        clamp_max_1: "f32[s4]" = torch.ops.aten.clamp_max.default(ceil_3, sub_7);  ceil_3 = sub_7 = None
        convert_element_type_7: "i64[s4]" = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.int64);  clamp_max_1 = None
        unsqueeze: "f32[s3, 1]" = torch.ops.aten.unsqueeze.default(mul_2, 1);  mul_2 = None
        unsqueeze_1: "i64[s3, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 1);  convert_element_type_4 = None
        unsqueeze_2: "i64[s3, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 1);  convert_element_type_5 = None
        _unsafe_index: "f32[1, s0, s3, s4]" = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, unsqueeze_1, convert_element_type_6])
        _unsafe_index_1: "f32[1, s0, s3, s4]" = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, unsqueeze_2, convert_element_type_6])
        _unsafe_index_2: "f32[1, s0, s3, s4]" = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, unsqueeze_1, convert_element_type_7])
        _unsafe_index_3: "f32[1, s0, s3, s4]" = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, unsqueeze_2, convert_element_type_7]);  arg3_1 = unsqueeze_2 = convert_element_type_7 = None
        sub_8: "f32[s3, 1]" = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        sub_9: "f32[s3, 1]" = torch.ops.aten.sub.Tensor(1.0, sub_8)
        sub_10: "f32[s4]" = torch.ops.aten.sub.Tensor(mul_3, convert_element_type_6);  mul_3 = convert_element_type_6 = None
        sub_11: "f32[s4]" = torch.ops.aten.sub.Tensor(1.0, sub_10)
        mul_4: "f32[1, s0, s3, s4]" = torch.ops.aten.mul.Tensor(_unsafe_index, sub_9);  _unsafe_index = None
        mul_5: "f32[1, s0, s3, s4]" = torch.ops.aten.mul.Tensor(_unsafe_index_1, sub_8);  _unsafe_index_1 = None
        add_2: "f32[1, s0, s3, s4]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        mul_6: "f32[1, s0, s3, s4]" = torch.ops.aten.mul.Tensor(_unsafe_index_2, sub_9);  _unsafe_index_2 = sub_9 = None
        mul_7: "f32[1, s0, s3, s4]" = torch.ops.aten.mul.Tensor(_unsafe_index_3, sub_8);  _unsafe_index_3 = sub_8 = None
        add_3: "f32[1, s0, s3, s4]" = torch.ops.aten.add.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
        mul_8: "f32[1, s0, s3, s4]" = torch.ops.aten.mul.Tensor(add_2, sub_11);  add_2 = sub_11 = None
        mul_9: "f32[1, s0, s3, s4]" = torch.ops.aten.mul.Tensor(add_3, sub_10);  add_3 = sub_10 = None
        add_4: "f32[1, s0, s3, s4]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        return (add_4,)
        