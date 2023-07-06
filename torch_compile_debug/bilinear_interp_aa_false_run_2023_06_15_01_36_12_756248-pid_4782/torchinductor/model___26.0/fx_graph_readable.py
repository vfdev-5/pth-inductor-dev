class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: f32[3, 345, 456]):
        # File: check_interpolate.py:40, code: img = img[None, ...]
        unsqueeze: f32[1, 3, 345, 456] = torch.ops.aten.unsqueeze.default(arg0_1, 0);  arg0_1 = None

        # File: check_interpolate.py:41, code: img = torch.nn.functional.interpolate(img, size=(224, 224), mode="bilinear", antialias=False)
        iota: i64[224] = torch.ops.prims.iota.default(224, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        convert_element_type: f64[224] = torch.ops.prims.convert_element_type.default(iota, torch.float64);  iota = None
        mul: f64[224] = torch.ops.aten.mul.Tensor(convert_element_type, 1);  convert_element_type = None
        add: f64[224] = torch.ops.aten.add.Tensor(mul, 0);  mul = None
        convert_element_type_1: f32[224] = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None

        iota_1: i64[224] = torch.ops.prims.iota.default(224, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        convert_element_type_2: f64[224] = torch.ops.prims.convert_element_type.default(iota_1, torch.float64);  iota_1 = None
        mul_1: f64[224] = torch.ops.aten.mul.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None
        add_1: f64[224] = torch.ops.aten.add.Tensor(mul_1, 0);  mul_1 = None
        convert_element_type_3: f32[224] = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None

        add_2: f32[224] = torch.ops.aten.add.Tensor(convert_element_type_1, 0.5);  convert_element_type_1 = None
        mul_2: f32[224] = torch.ops.aten.mul.Tensor(add_2, 1.5401785714285714);  add_2 = None
        sub: f32[224] = torch.ops.aten.sub.Tensor(mul_2, 0.5);  mul_2 = None
        clamp_min: f32[224] = torch.ops.aten.clamp_min.default(sub, 0.0);  sub = None

        add_3: f32[224] = torch.ops.aten.add.Tensor(convert_element_type_3, 0.5);  convert_element_type_3 = None
        mul_3: f32[224] = torch.ops.aten.mul.Tensor(add_3, 2.0357142857142856);  add_3 = None
        sub_1: f32[224] = torch.ops.aten.sub.Tensor(mul_3, 0.5);  mul_3 = None
        clamp_min_1: f32[224] = torch.ops.aten.clamp_min.default(sub_1, 0.0);  sub_1 = None


        convert_element_type_4: i64[224] = torch.ops.prims.convert_element_type.default(clamp_min, torch.int64)
        ceil: f32[224] = torch.ops.aten.ceil.default(clamp_min)
        clamp_max: f32[224] = torch.ops.aten.clamp_max.default(ceil, 344);  ceil = None
        convert_element_type_5: i64[224] = torch.ops.prims.convert_element_type.default(clamp_max, torch.int64);  clamp_max = None
        convert_element_type_6: i64[224] = torch.ops.prims.convert_element_type.default(clamp_min_1, torch.int64)
        ceil_1: f32[224] = torch.ops.aten.ceil.default(clamp_min_1)
        clamp_max_1: f32[224] = torch.ops.aten.clamp_max.default(ceil_1, 455);  ceil_1 = None
        convert_element_type_7: i64[224] = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.int64);  clamp_max_1 = None
        unsqueeze_1: f32[224, 1] = torch.ops.aten.unsqueeze.default(clamp_min, 1);  clamp_min = None
        unsqueeze_2: i64[224, 1] = torch.ops.aten.unsqueeze.default(convert_element_type_4, 1);  convert_element_type_4 = None
        unsqueeze_3: i64[224, 1] = torch.ops.aten.unsqueeze.default(convert_element_type_5, 1);  convert_element_type_5 = None
        _unsafe_index: f32[1, 3, 224, 224] = torch.ops.aten._unsafe_index.Tensor(unsqueeze, [None, None, unsqueeze_2, convert_element_type_6])
        _unsafe_index_1: f32[1, 3, 224, 224] = torch.ops.aten._unsafe_index.Tensor(unsqueeze, [None, None, unsqueeze_3, convert_element_type_6])
        _unsafe_index_2: f32[1, 3, 224, 224] = torch.ops.aten._unsafe_index.Tensor(unsqueeze, [None, None, unsqueeze_2, convert_element_type_7])
        _unsafe_index_3: f32[1, 3, 224, 224] = torch.ops.aten._unsafe_index.Tensor(unsqueeze, [None, None, unsqueeze_3, convert_element_type_7]);  unsqueeze = unsqueeze_3 = convert_element_type_7 = None
        sub_2: f32[224, 1] = torch.ops.aten.sub.Tensor(unsqueeze_1, unsqueeze_2);  unsqueeze_1 = unsqueeze_2 = None
        sub_3: f32[224, 1] = torch.ops.aten.sub.Tensor(1.0, sub_2)
        sub_4: f32[224] = torch.ops.aten.sub.Tensor(clamp_min_1, convert_element_type_6);  clamp_min_1 = convert_element_type_6 = None
        sub_5: f32[224] = torch.ops.aten.sub.Tensor(1.0, sub_4)
        mul_4: f32[1, 3, 224, 224] = torch.ops.aten.mul.Tensor(_unsafe_index, sub_3);  _unsafe_index = None
        mul_5: f32[1, 3, 224, 224] = torch.ops.aten.mul.Tensor(_unsafe_index_1, sub_2);  _unsafe_index_1 = None
        add_4: f32[1, 3, 224, 224] = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        mul_6: f32[1, 3, 224, 224] = torch.ops.aten.mul.Tensor(_unsafe_index_2, sub_3);  _unsafe_index_2 = sub_3 = None
        mul_7: f32[1, 3, 224, 224] = torch.ops.aten.mul.Tensor(_unsafe_index_3, sub_2);  _unsafe_index_3 = sub_2 = None
        add_5: f32[1, 3, 224, 224] = torch.ops.aten.add.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
        mul_8: f32[1, 3, 224, 224] = torch.ops.aten.mul.Tensor(add_4, sub_5);  add_4 = sub_5 = None
        mul_9: f32[1, 3, 224, 224] = torch.ops.aten.mul.Tensor(add_5, sub_4);  add_5 = sub_4 = None
        add_6: f32[1, 3, 224, 224] = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        return (add_6,)
