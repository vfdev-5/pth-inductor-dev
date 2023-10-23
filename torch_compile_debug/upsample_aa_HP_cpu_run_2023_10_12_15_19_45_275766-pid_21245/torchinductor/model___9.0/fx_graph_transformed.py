class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: f32[4, 3, 345, 456]):
        # File: check_interpolate_bilinear_aa.py:18, code: img = torch.nn.functional.interpolate(img, size=(345, 272), mode="bilinear", antialias=True)
        iota_1: i64[5] = torch.ops.prims.iota.default(5, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        iota: i64[272] = torch.ops.prims.iota.default(272, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        add: f32[272] = torch.ops.aten.add.Tensor(iota, 0.5);  iota = None
        mul: f32[272] = torch.ops.aten.mul.Tensor(add, 1.6764705882352942);  add = None
        add_2: f32[272] = torch.ops.aten.add.Tensor(mul, 1.6764705882352942)
        add_3: f32[272] = torch.ops.aten.add.Tensor(add_2, 0.5);  add_2 = None
        convert_element_type_1: i64[272] = torch.ops.prims.convert_element_type.default(add_3, torch.int64);  add_3 = None
        clamp_max: i64[272] = torch.ops.aten.clamp_max.default(convert_element_type_1, 456);  convert_element_type_1 = None
        sub: f32[272] = torch.ops.aten.sub.Tensor(mul, 1.6764705882352942)
        add_1: f32[272] = torch.ops.aten.add.Tensor(sub, 0.5);  sub = None
        convert_element_type: i64[272] = torch.ops.prims.convert_element_type.default(add_1, torch.int64);  add_1 = None
        clamp_min: i64[272] = torch.ops.aten.clamp_min.default(convert_element_type, 0);  convert_element_type = None
        sub_1: i64[272] = torch.ops.aten.sub.Tensor(clamp_max, clamp_min);  clamp_max = None
        clamp_max_1: i64[272] = torch.ops.aten.clamp_max.default(sub_1, 5);  sub_1 = None
        view_2: i64[272, 1] = torch.ops.aten.reshape.default(clamp_max_1, [-1, 1]);  clamp_max_1 = None
        lt: b8[272, 5] = torch.ops.aten.lt.Tensor(iota_1, view_2);  view_2 = None
        view_1: i64[272, 1] = torch.ops.aten.reshape.default(clamp_min, [-1, 1]);  clamp_min = None
        add_4: i64[272, 5] = torch.ops.aten.add.Tensor(iota_1, view_1);  iota_1 = None
        view: f32[272, 1] = torch.ops.aten.reshape.default(mul, [-1, 1]);  mul = None
        sub_2: f32[272, 5] = torch.ops.aten.sub.Tensor(add_4, view);  add_4 = view = None
        add_5: f32[272, 5] = torch.ops.aten.add.Tensor(sub_2, 0.5);  sub_2 = None
        mul_1: f32[272, 5] = torch.ops.aten.mul.Tensor(add_5, 0.5964912280701754);  add_5 = None
        abs_1: f32[272, 5] = torch.ops.aten.abs.default(mul_1);  mul_1 = None
        clamp_max_2: f32[272, 5] = torch.ops.aten.clamp_max.default(abs_1, 1.0);  abs_1 = None
        sub_3: f32[272, 5] = torch.ops.aten.sub.Tensor(1.0, clamp_max_2);  clamp_max_2 = None
        full_default: f32[] = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where: f32[272, 5] = torch.ops.aten.where.self(lt, sub_3, full_default);  lt = sub_3 = full_default = None
        sum_1: f32[272] = torch.ops.aten.sum.dim_IntList(where, [-1])
        unsqueeze: f32[272, 1] = torch.ops.aten.unsqueeze.default(sum_1, -1);  sum_1 = None
        div: f32[272, 5] = torch.ops.aten.div.Tensor(where, unsqueeze);  where = unsqueeze = None
        view_3: i64[272] = torch.ops.aten.reshape.default(view_1, [-1]);  view_1 = None
        unsqueeze_1: i64[272, 1] = torch.ops.aten.unsqueeze.default(view_3, -1);  view_3 = None
        iota_2: i64[5] = torch.ops.prims.iota.default(5, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        add_6: i64[272, 5] = torch.ops.aten.add.Tensor(unsqueeze_1, iota_2);  unsqueeze_1 = iota_2 = None
        clamp_max_3: i64[272, 5] = torch.ops.aten.clamp_max.default(add_6, 455);  add_6 = None
        index: f32[4, 3, 345, 272, 5] = torch.ops.aten.index.Tensor(arg0_1, [None, None, None, clamp_max_3]);  arg0_1 = clamp_max_3 = None
        mul_2: f32[4, 3, 345, 272, 5] = torch.ops.aten.mul.Tensor(div, index);  div = index = None
        sum_2: f32[4, 3, 345, 272] = torch.ops.aten.sum.dim_IntList(mul_2, [-1]);  mul_2 = None
        return (sum_2,)
        