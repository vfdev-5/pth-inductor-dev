class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "Sym(s0)", arg1_1: "Sym(s1)", arg2_1: "Sym(s2)", arg3_1: "f32[1, s0, s1, s2]", arg4_1: "Sym(1234)", arg5_1: "Sym(1345)"):
        # File: repro_interpolate_bilinear_compile_error.py:14, code: img = F.interpolate(img, size=osize, mode="bilinear", antialias=False, align_corners=align_corners)
        iota: "i64[1234]" = torch.ops.prims.iota.default(1234, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        convert_element_type: "f32[1234]" = torch.ops.prims.convert_element_type.default(iota, torch.float32);  iota = None
        add: "f32[1234]" = torch.ops.aten.add.Tensor(convert_element_type, 0.5);  convert_element_type = None
        sym_size_int: "Sym(s1)" = torch.ops.aten.sym_size.int(arg3_1, 2)
        truediv: "Sym(s1/1234)" = sym_size_int / 1234
        mul: "f32[1234]" = torch.ops.aten.mul.Tensor(add, truediv);  add = truediv = None
        sub: "f32[1234]" = torch.ops.aten.sub.Tensor(mul, 0.5);  mul = None
        clamp_min: "f32[1234]" = torch.ops.aten.clamp_min.default(sub, 0.0);  sub = None
        view: "f32[1234, 1]" = torch.ops.aten.view.default(clamp_min, [1234, 1]);  clamp_min = None
        convert_element_type_1: "i64[1234, 1]" = torch.ops.prims.convert_element_type.default(view, torch.int64)
        sub_1: "Sym(s1 - 1)" = sym_size_int - 1;  sym_size_int = None
        lt: "b8[1234, 1]" = torch.ops.aten.lt.Scalar(convert_element_type_1, sub_1);  sub_1 = None
        add_1: "i64[1234, 1]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1)
        where: "i64[1234, 1]" = torch.ops.aten.where.self(lt, add_1, convert_element_type_1);  lt = add_1 = None
        iota_1: "i64[1345]" = torch.ops.prims.iota.default(1345, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        convert_element_type_2: "f32[1345]" = torch.ops.prims.convert_element_type.default(iota_1, torch.float32);  iota_1 = None
        add_2: "f32[1345]" = torch.ops.aten.add.Tensor(convert_element_type_2, 0.5);  convert_element_type_2 = None
        sym_size_int_1: "Sym(s2)" = torch.ops.aten.sym_size.int(arg3_1, 3)
        truediv_1: "Sym(s2/1345)" = sym_size_int_1 / 1345
        mul_1: "f32[1345]" = torch.ops.aten.mul.Tensor(add_2, truediv_1);  add_2 = truediv_1 = None
        sub_2: "f32[1345]" = torch.ops.aten.sub.Tensor(mul_1, 0.5);  mul_1 = None
        clamp_min_1: "f32[1345]" = torch.ops.aten.clamp_min.default(sub_2, 0.0);  sub_2 = None
        convert_element_type_3: "i64[1345]" = torch.ops.prims.convert_element_type.default(clamp_min_1, torch.int64)
        sub_3: "Sym(s2 - 1)" = sym_size_int_1 - 1;  sym_size_int_1 = None
        lt_1: "b8[1345]" = torch.ops.aten.lt.Scalar(convert_element_type_3, sub_3);  sub_3 = None
        add_3: "i64[1345]" = torch.ops.aten.add.Tensor(convert_element_type_3, 1)
        where_1: "i64[1345]" = torch.ops.aten.where.self(lt_1, add_3, convert_element_type_3);  lt_1 = add_3 = None
        _unsafe_index: "f32[1, s0, 1234, 1345]" = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, convert_element_type_1, convert_element_type_3])
        _unsafe_index_1: "f32[1, s0, 1234, 1345]" = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, convert_element_type_1, where_1])
        _unsafe_index_2: "f32[1, s0, 1234, 1345]" = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, where, convert_element_type_3])
        _unsafe_index_3: "f32[1, s0, 1234, 1345]" = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, where, where_1]);  arg3_1 = where = where_1 = None
        sub_4: "f32[1345]" = torch.ops.aten.sub.Tensor(clamp_min_1, convert_element_type_3);  clamp_min_1 = convert_element_type_3 = None
        clamp_min_2: "f32[1345]" = torch.ops.aten.clamp_min.default(sub_4, 0.0);  sub_4 = None
        clamp_max: "f32[1345]" = torch.ops.aten.clamp_max.default(clamp_min_2, 1.0);  clamp_min_2 = None
        sub_5: "f32[1, s0, 1234, 1345]" = torch.ops.aten.sub.Tensor(_unsafe_index_1, _unsafe_index);  _unsafe_index_1 = None
        mul_2: "f32[1, s0, 1234, 1345]" = torch.ops.aten.mul.Tensor(sub_5, clamp_max);  sub_5 = None
        add_4: "f32[1, s0, 1234, 1345]" = torch.ops.aten.add.Tensor(_unsafe_index, mul_2);  _unsafe_index = mul_2 = None
        sub_6: "f32[1, s0, 1234, 1345]" = torch.ops.aten.sub.Tensor(_unsafe_index_3, _unsafe_index_2);  _unsafe_index_3 = None
        mul_3: "f32[1, s0, 1234, 1345]" = torch.ops.aten.mul.Tensor(sub_6, clamp_max);  sub_6 = clamp_max = None
        add_5: "f32[1, s0, 1234, 1345]" = torch.ops.aten.add.Tensor(_unsafe_index_2, mul_3);  _unsafe_index_2 = mul_3 = None
        sub_7: "f32[1234, 1]" = torch.ops.aten.sub.Tensor(view, convert_element_type_1);  view = convert_element_type_1 = None
        clamp_min_3: "f32[1234, 1]" = torch.ops.aten.clamp_min.default(sub_7, 0.0);  sub_7 = None
        clamp_max_1: "f32[1234, 1]" = torch.ops.aten.clamp_max.default(clamp_min_3, 1.0);  clamp_min_3 = None
        sub_8: "f32[1, s0, 1234, 1345]" = torch.ops.aten.sub.Tensor(add_5, add_4);  add_5 = None
        mul_4: "f32[1, s0, 1234, 1345]" = torch.ops.aten.mul.Tensor(sub_8, clamp_max_1);  sub_8 = clamp_max_1 = None
        add_6: "f32[1, s0, 1234, 1345]" = torch.ops.aten.add.Tensor(add_4, mul_4);  add_4 = mul_4 = None
        return (add_6,)
        