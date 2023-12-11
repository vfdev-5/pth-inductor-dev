class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "u8[2, 3, 500, 500]"):
        # File: check_interpolate_bilinear.py:12, code: img = torch.nn.functional.interpolate(img, size=osize, mode="bilinear", antialias=False, align_corners=False)
        iota: "i32[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int32, device = device(type='cpu'), requires_grad = False)
        iota_1: "i32[256]" = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int32, device = device(type='cpu'), requires_grad = False)
        add: "f32[256]" = torch.ops.aten.add.Tensor(iota_1, 0.5);  iota_1 = None
        mul: "f32[256]" = torch.ops.aten.mul.Tensor(add, 1.953125);  add = None
        sub: "f32[256]" = torch.ops.aten.sub.Tensor(mul, 0.5);  mul = None
        clamp_min: "f32[256]" = torch.ops.aten.clamp_min.default(sub, 0.0);  sub = None
        add_1: "f32[256]" = torch.ops.aten.add.Tensor(iota, 0.5);  iota = None
        mul_1: "f32[256]" = torch.ops.aten.mul.Tensor(add_1, 1.953125);  add_1 = None
        sub_1: "f32[256]" = torch.ops.aten.sub.Tensor(mul_1, 0.5);  mul_1 = None
        clamp_min_1: "f32[256]" = torch.ops.aten.clamp_min.default(sub_1, 0.0);  sub_1 = None
        convert_element_type: "i32[256]" = torch.ops.prims.convert_element_type.default(clamp_min, torch.int32)
        convert_element_type_1: "i32[256]" = torch.ops.prims.convert_element_type.default(clamp_min_1, torch.int32)
        iota_2: "i32[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int32, device = device(type='cpu'), requires_grad = False)
        iota_3: "i32[2]" = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int32, device = device(type='cpu'), requires_grad = False)
        unsqueeze: "i32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1)
        add_2: "i32[256, 2]" = torch.ops.aten.add.Tensor(unsqueeze, iota_2);  unsqueeze = iota_2 = None
        clamp_max: "i32[256, 2]" = torch.ops.aten.clamp_max.default(add_2, 499);  add_2 = None
        unsqueeze_1: "i32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_1, -1)
        add_3: "i32[256, 2]" = torch.ops.aten.add.Tensor(unsqueeze_1, iota_3);  unsqueeze_1 = iota_3 = None
        clamp_max_1: "i32[256, 2]" = torch.ops.aten.clamp_max.default(add_3, 499);  add_3 = None
        sub_2: "f32[256]" = torch.ops.aten.sub.Tensor(clamp_min, convert_element_type);  clamp_min = convert_element_type = None
        clamp_min_2: "f32[256]" = torch.ops.aten.clamp_min.default(sub_2, 0.0);  sub_2 = None
        clamp_max_2: "f32[256]" = torch.ops.aten.clamp_max.default(clamp_min_2, 1.0);  clamp_min_2 = None
        sub_3: "f32[256]" = torch.ops.aten.sub.Tensor(1.0, clamp_max_2)
        unsqueeze_2: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(sub_3, 1);  sub_3 = None
        unsqueeze_3: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_2, 1);  clamp_max_2 = None
        cat: "f32[256, 2]" = torch.ops.aten.cat.default([unsqueeze_2, unsqueeze_3], -1);  unsqueeze_2 = unsqueeze_3 = None
        sub_4: "f32[256]" = torch.ops.aten.sub.Tensor(clamp_min_1, convert_element_type_1);  clamp_min_1 = convert_element_type_1 = None
        clamp_min_3: "f32[256]" = torch.ops.aten.clamp_min.default(sub_4, 0.0);  sub_4 = None
        clamp_max_3: "f32[256]" = torch.ops.aten.clamp_max.default(clamp_min_3, 1.0);  clamp_min_3 = None
        sub_5: "f32[256]" = torch.ops.aten.sub.Tensor(1.0, clamp_max_3)
        unsqueeze_4: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(sub_5, 1);  sub_5 = None
        unsqueeze_5: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(clamp_max_3, 1);  clamp_max_3 = None
        cat_1: "f32[256, 2]" = torch.ops.aten.cat.default([unsqueeze_4, unsqueeze_5], -1);  unsqueeze_4 = unsqueeze_5 = None
        view: "i32[256, 2, 1, 1]" = torch.ops.aten.view.default(clamp_max_1, [256, 2, 1, 1]);  clamp_max_1 = None
        _unsafe_index: "u8[2, 3, 256, 2, 256, 2]" = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, view, clamp_max]);  arg0_1 = view = clamp_max = None
        view_1: "f32[256, 2, 1, 1]" = torch.ops.aten.view.default(cat_1, [256, 2, 1, 1]);  cat_1 = None
        view_2: "f32[1, 256, 2]" = torch.ops.aten.view.default(cat, [1, 256, 2]);  cat = None
        mul_2: "f32[2, 3, 256, 2, 256, 2]" = torch.ops.aten.mul.Tensor(view_2, _unsafe_index);  view_2 = _unsafe_index = None
        mul_3: "f32[2, 3, 256, 2, 256, 2]" = torch.ops.aten.mul.Tensor(view_1, mul_2);  view_1 = mul_2 = None
        sum_1: "f32[2, 3, 256, 256]" = torch.ops.aten.sum.dim_IntList(mul_3, [-1, -3]);  mul_3 = None
        round_1: "f32[2, 3, 256, 256]" = torch.ops.aten.round.default(sum_1);  sum_1 = None
        convert_element_type_2: "u8[2, 3, 256, 256]" = torch.ops.prims.convert_element_type.default(round_1, torch.uint8);  round_1 = None
        return (convert_element_type_2,)
        