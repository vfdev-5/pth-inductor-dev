class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: f32[2, 3, 500, 400]):
        # File: check_interpolate_nearest.py:12, code: img = torch.nn.functional.interpolate(img, size=(256, 256), mode="nearest")
        iota: i64[256] = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type: f32[256] = torch.ops.prims.convert_element_type.default(iota, torch.float32);  iota = None
        mul: f32[256] = torch.ops.aten.mul.Tensor(convert_element_type, 1);  convert_element_type = None
        add: f32[256] = torch.ops.aten.add.Tensor(mul, 0);  mul = None
        mul_1: f32[256] = torch.ops.aten.mul.Tensor(add, 1.953125);  add = None
        convert_element_type_1: i64[256] = torch.ops.prims.convert_element_type.default(mul_1, torch.int64);  mul_1 = None
        unsqueeze: i64[256, 1] = torch.ops.aten.unsqueeze.default(convert_element_type_1, -1);  convert_element_type_1 = None
        iota_1: i64[256] = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_2: f32[256] = torch.ops.prims.convert_element_type.default(iota_1, torch.float32);  iota_1 = None
        mul_2: f32[256] = torch.ops.aten.mul.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None
        add_1: f32[256] = torch.ops.aten.add.Tensor(mul_2, 0);  mul_2 = None
        mul_3: f32[256] = torch.ops.aten.mul.Tensor(add_1, 1.5625);  add_1 = None
        convert_element_type_3: i64[256] = torch.ops.prims.convert_element_type.default(mul_3, torch.int64);  mul_3 = None
        _unsafe_index: f32[2, 3, 256, 256] = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, unsqueeze, convert_element_type_3]);  arg0_1 = unsqueeze = convert_element_type_3 = None
        return (_unsafe_index,)
        