class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "u8[2, 3, 500, 500]"):
        # File: check_interpolate_bilinear.py:12, code: img = torch.nn.functional.interpolate(img, size=osize, mode="bilinear", antialias=False, align_corners=False)
        iota: "u8[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.uint8, device = device(type='cpu'), requires_grad = False)
        add: "f32[256]" = torch.ops.aten.add.Tensor(iota, 0.5);  iota = None
        mul: "f32[256]" = torch.ops.aten.mul.Tensor(add, 1.953125);  add = None
        sub: "f32[256]" = torch.ops.aten.sub.Tensor(mul, 0.5);  mul = None
        clamp_min: "f32[256]" = torch.ops.aten.clamp_min.default(sub, 0.0);  sub = None
        convert_element_type: "i64[256]" = torch.ops.prims.convert_element_type.default(clamp_min, torch.int64)
        unsqueeze_1: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, 1);  convert_element_type = None
        iota_1: "u8[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.uint8, device = device(type='cpu'), requires_grad = False)
        add_1: "f32[256]" = torch.ops.aten.add.Tensor(iota_1, 0.5);  iota_1 = None
        mul_1: "f32[256]" = torch.ops.aten.mul.Tensor(add_1, 1.953125);  add_1 = None
        sub_1: "f32[256]" = torch.ops.aten.sub.Tensor(mul_1, 0.5);  mul_1 = None
        clamp_min_1: "f32[256]" = torch.ops.aten.clamp_min.default(sub_1, 0.0);  sub_1 = None
        convert_element_type_2: "i64[256]" = torch.ops.prims.convert_element_type.default(clamp_min_1, torch.int64)
        _unsafe_index: "u8[2, 3, 256, 256]" = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, unsqueeze_1, convert_element_type_2])
        unsqueeze: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(clamp_min, 1)
        sub_2: "f32[256, 1]" = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = None
        sub_3: "f32[256, 1]" = torch.ops.aten.sub.Tensor(1.0, sub_2)
        mul_2: "f32[2, 3, 256, 256]" = torch.ops.aten.mul.Tensor(_unsafe_index, sub_3);  _unsafe_index = None
        ceil: "f32[256]" = torch.ops.aten.ceil.default(clamp_min);  clamp_min = None
        clamp_max: "f32[256]" = torch.ops.aten.clamp_max.default(ceil, 499);  ceil = None
        convert_element_type_1: "i64[256]" = torch.ops.prims.convert_element_type.default(clamp_max, torch.int64);  clamp_max = None
        unsqueeze_2: "i64[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1);  convert_element_type_1 = None
        _unsafe_index_1: "u8[2, 3, 256, 256]" = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, unsqueeze_2, convert_element_type_2])
        mul_3: "f32[2, 3, 256, 256]" = torch.ops.aten.mul.Tensor(_unsafe_index_1, sub_2);  _unsafe_index_1 = None
        add_2: "f32[2, 3, 256, 256]" = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        sub_4: "f32[256]" = torch.ops.aten.sub.Tensor(clamp_min_1, convert_element_type_2);  convert_element_type_2 = None
        sub_5: "f32[256]" = torch.ops.aten.sub.Tensor(1.0, sub_4)
        mul_6: "f32[2, 3, 256, 256]" = torch.ops.aten.mul.Tensor(add_2, sub_5);  add_2 = sub_5 = None
        ceil_1: "f32[256]" = torch.ops.aten.ceil.default(clamp_min_1);  clamp_min_1 = None
        clamp_max_1: "f32[256]" = torch.ops.aten.clamp_max.default(ceil_1, 499);  ceil_1 = None
        convert_element_type_3: "i64[256]" = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.int64);  clamp_max_1 = None
        _unsafe_index_2: "u8[2, 3, 256, 256]" = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, unsqueeze_1, convert_element_type_3]);  unsqueeze_1 = None
        mul_4: "f32[2, 3, 256, 256]" = torch.ops.aten.mul.Tensor(_unsafe_index_2, sub_3);  _unsafe_index_2 = sub_3 = None
        _unsafe_index_3: "u8[2, 3, 256, 256]" = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, unsqueeze_2, convert_element_type_3]);  arg0_1 = unsqueeze_2 = convert_element_type_3 = None
        mul_5: "f32[2, 3, 256, 256]" = torch.ops.aten.mul.Tensor(_unsafe_index_3, sub_2);  _unsafe_index_3 = sub_2 = None
        add_3: "f32[2, 3, 256, 256]" = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        mul_7: "f32[2, 3, 256, 256]" = torch.ops.aten.mul.Tensor(add_3, sub_4);  add_3 = sub_4 = None
        add_4: "f32[2, 3, 256, 256]" = torch.ops.aten.add.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
        convert_element_type_4: "u8[2, 3, 256, 256]" = torch.ops.prims.convert_element_type.default(add_4, torch.uint8);  add_4 = None
        return (convert_element_type_4,)
        