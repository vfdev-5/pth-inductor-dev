class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s0)", arg1_1: "Sym(s1)", arg2_1: "Sym(s2)", arg3_1: "u8[s0, s1, s2, s2]", arg4_1: "Sym(s3)", arg5_1: "Sym(s4)"):
        # File: check_interpolate_bilinear.py:12, code: img = torch.nn.functional.interpolate(img, size=osize, mode="bilinear", antialias=False, align_corners=False)
        sub: "Sym(s3)" = arg4_1 - 0
        truediv: "Sym(s3)" = sub / 1;  sub = None
        ceil: "Sym(s3)" = math_ceil(truediv);  truediv = None
        iota: "u8[s3]" = torch.ops.prims.iota.default(ceil, start = 0, step = 1, dtype = torch.uint8, device = device(type='cpu'), requires_grad = False);  ceil = None
        sub_1: "Sym(s4)" = arg5_1 - 0
        truediv_1: "Sym(s4)" = sub_1 / 1;  sub_1 = None
        ceil_1: "Sym(s4)" = math_ceil(truediv_1);  truediv_1 = None
        iota_1: "u8[s4]" = torch.ops.prims.iota.default(ceil_1, start = 0, step = 1, dtype = torch.uint8, device = device(type='cpu'), requires_grad = False);  ceil_1 = None
        add: "f32[s3]" = torch.ops.aten.add.Tensor(iota, 0.5);  iota = None
        truediv_2: "Sym(s2/s3)" = arg2_1 / arg4_1;  arg4_1 = None
        mul: "f32[s3]" = torch.ops.aten.mul.Tensor(add, truediv_2);  add = truediv_2 = None
        sub_2: "f32[s3]" = torch.ops.aten.sub.Tensor(mul, 0.5);  mul = None
        clamp_min: "f32[s3]" = torch.ops.aten.clamp_min.default(sub_2, 0.0);  sub_2 = None
        add_1: "f32[s4]" = torch.ops.aten.add.Tensor(iota_1, 0.5);  iota_1 = None
        sym_size_int: "Sym(s2)" = torch.ops.aten.sym_size.int(arg3_1, 3)
        truediv_3: "Sym(s2/s4)" = sym_size_int / arg5_1;  arg5_1 = None
        mul_1: "f32[s4]" = torch.ops.aten.mul.Tensor(add_1, truediv_3);  add_1 = truediv_3 = None
        sub_3: "f32[s4]" = torch.ops.aten.sub.Tensor(mul_1, 0.5);  mul_1 = None
        clamp_min_1: "f32[s4]" = torch.ops.aten.clamp_min.default(sub_3, 0.0);  sub_3 = None
        convert_element_type: "i64[s3]" = torch.ops.prims.convert_element_type.default(clamp_min, torch.int64)
        ceil_2: "f32[s3]" = torch.ops.aten.ceil.default(clamp_min)
        sub_4: "Sym(s2 - 1)" = arg2_1 - 1;  arg2_1 = None
        clamp_max: "f32[s3]" = torch.ops.aten.clamp_max.default(ceil_2, sub_4);  ceil_2 = sub_4 = None
        convert_element_type_1: "i64[s3]" = torch.ops.prims.convert_element_type.default(clamp_max, torch.int64);  clamp_max = None
        convert_element_type_2: "i64[s4]" = torch.ops.prims.convert_element_type.default(clamp_min_1, torch.int64)
        ceil_3: "f32[s4]" = torch.ops.aten.ceil.default(clamp_min_1)
        sub_5: "Sym(s2 - 1)" = sym_size_int - 1;  sym_size_int = None
        clamp_max_1: "f32[s4]" = torch.ops.aten.clamp_max.default(ceil_3, sub_5);  ceil_3 = sub_5 = None
        convert_element_type_3: "i64[s4]" = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.int64);  clamp_max_1 = None
        unsqueeze: "f32[s3, 1]" = torch.ops.aten.unsqueeze.default(clamp_min, 1);  clamp_min = None
        unsqueeze_1: "i64[s3, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, 1);  convert_element_type = None
        unsqueeze_2: "i64[s3, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1);  convert_element_type_1 = None
        _unsafe_index: "u8[s0, s1, s3, s4]" = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, unsqueeze_1, convert_element_type_2])
        _unsafe_index_1: "u8[s0, s1, s3, s4]" = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, unsqueeze_2, convert_element_type_2])
        _unsafe_index_2: "u8[s0, s1, s3, s4]" = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, unsqueeze_1, convert_element_type_3])
        _unsafe_index_3: "u8[s0, s1, s3, s4]" = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, unsqueeze_2, convert_element_type_3]);  arg3_1 = unsqueeze_2 = convert_element_type_3 = None
        sub_6: "f32[s3, 1]" = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        sub_7: "f32[s3, 1]" = torch.ops.aten.sub.Tensor(1.0, sub_6)
        sub_8: "f32[s4]" = torch.ops.aten.sub.Tensor(clamp_min_1, convert_element_type_2);  clamp_min_1 = convert_element_type_2 = None
        sub_9: "f32[s4]" = torch.ops.aten.sub.Tensor(1.0, sub_8)
        mul_2: "f32[s0, s1, s3, s4]" = torch.ops.aten.mul.Tensor(_unsafe_index, sub_7);  _unsafe_index = None
        mul_3: "f32[s0, s1, s3, s4]" = torch.ops.aten.mul.Tensor(_unsafe_index_1, sub_6);  _unsafe_index_1 = None
        add_2: "f32[s0, s1, s3, s4]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        mul_4: "f32[s0, s1, s3, s4]" = torch.ops.aten.mul.Tensor(_unsafe_index_2, sub_7);  _unsafe_index_2 = sub_7 = None
        mul_5: "f32[s0, s1, s3, s4]" = torch.ops.aten.mul.Tensor(_unsafe_index_3, sub_6);  _unsafe_index_3 = sub_6 = None
        add_3: "f32[s0, s1, s3, s4]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        mul_6: "f32[s0, s1, s3, s4]" = torch.ops.aten.mul.Tensor(add_2, sub_9);  add_2 = sub_9 = None
        mul_7: "f32[s0, s1, s3, s4]" = torch.ops.aten.mul.Tensor(add_3, sub_8);  add_3 = sub_8 = None
        add_4: "f32[s0, s1, s3, s4]" = torch.ops.aten.add.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
        convert_element_type_4: "u8[s0, s1, s3, s4]" = torch.ops.prims.convert_element_type.default(add_4, torch.uint8);  add_4 = None
        return (convert_element_type_4,)
        