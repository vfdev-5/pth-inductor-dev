class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[3, 640, 4, 4]"):
        # File: check_issue_upsample_nearest.py:12 in forward, code: return F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        iota: "i64[8]" = torch.ops.prims.iota.default(8, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        convert_element_type: "f64[8]" = torch.ops.prims.convert_element_type.default(iota, torch.float64);  iota = None
        mul: "f64[8]" = torch.ops.aten.mul.Tensor(convert_element_type, 1);  convert_element_type = None
        add: "f64[8]" = torch.ops.aten.add.Tensor(mul, 0);  mul = None
        convert_element_type_1: "f32[8]" = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        add_1: "f32[8]" = torch.ops.aten.add.Tensor(convert_element_type_1, 0.0);  convert_element_type_1 = None
        mul_1: "f32[8]" = torch.ops.aten.mul.Tensor(add_1, 0.5);  add_1 = None
        convert_element_type_2: "i64[8]" = torch.ops.prims.convert_element_type.default(mul_1, torch.int64);  mul_1 = None
        unsqueeze: "i64[8, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_2, -1);  convert_element_type_2 = None
        iota_1: "i64[8]" = torch.ops.prims.iota.default(8, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        convert_element_type_3: "f64[8]" = torch.ops.prims.convert_element_type.default(iota_1, torch.float64);  iota_1 = None
        mul_2: "f64[8]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1);  convert_element_type_3 = None
        add_2: "f64[8]" = torch.ops.aten.add.Tensor(mul_2, 0);  mul_2 = None
        convert_element_type_4: "f32[8]" = torch.ops.prims.convert_element_type.default(add_2, torch.float32);  add_2 = None
        add_3: "f32[8]" = torch.ops.aten.add.Tensor(convert_element_type_4, 0.0);  convert_element_type_4 = None
        mul_3: "f32[8]" = torch.ops.aten.mul.Tensor(add_3, 0.5);  add_3 = None
        convert_element_type_5: "i64[8]" = torch.ops.prims.convert_element_type.default(mul_3, torch.int64);  mul_3 = None
        _unsafe_index: "f32[3, 640, 8, 8]" = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, unsqueeze, convert_element_type_5]);  arg0_1 = unsqueeze = convert_element_type_5 = None
        return (_unsafe_index,)
        