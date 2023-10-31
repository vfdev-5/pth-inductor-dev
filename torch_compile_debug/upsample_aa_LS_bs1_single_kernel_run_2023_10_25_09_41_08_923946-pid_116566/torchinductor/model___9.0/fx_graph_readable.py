class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: f32[1, 3, 3456, 4567]):
        # File: check_interpolate_bilinear_aa.py:30, code: img = torch.nn.functional.interpolate(img, size=osize, mode="bilinear", antialias=True)
        iota: i32[3456] = torch.ops.prims.iota.default(3456, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        add: f32[3456] = torch.ops.aten.add.Tensor(iota, 0.5);  iota = None
        mul: f32[3456] = torch.ops.aten.mul.Tensor(add, 1.3214699074074074);  add = None
        sub: f32[3456] = torch.ops.aten.sub.Tensor(mul, 1.3214699074074074)
        add_1: f32[3456] = torch.ops.aten.add.Tensor(sub, 0.5);  sub = None
        convert_element_type: i32[3456] = torch.ops.prims.convert_element_type.default(add_1, torch.int32);  add_1 = None
        clamp_min: i32[3456] = torch.ops.aten.clamp_min.default(convert_element_type, 0);  convert_element_type = None
        add_2: f32[3456] = torch.ops.aten.add.Tensor(mul, 1.3214699074074074)
        add_3: f32[3456] = torch.ops.aten.add.Tensor(add_2, 0.5);  add_2 = None
        convert_element_type_1: i32[3456] = torch.ops.prims.convert_element_type.default(add_3, torch.int32);  add_3 = None
        clamp_max: i32[3456] = torch.ops.aten.clamp_max.default(convert_element_type_1, 4567);  convert_element_type_1 = None
        sub_1: i32[3456] = torch.ops.aten.sub.Tensor(clamp_max, clamp_min);  clamp_max = None
        clamp_max_1: i32[3456] = torch.ops.aten.clamp_max.default(sub_1, 5);  sub_1 = None
        iota_1: i32[5] = torch.ops.prims.iota.default(5, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        view: f32[3456, 1] = torch.ops.aten.view.default(mul, [-1, 1]);  mul = None
        view_1: i32[3456, 1] = torch.ops.aten.view.default(clamp_min, [-1, 1]);  clamp_min = None
        view_2: i32[3456, 1] = torch.ops.aten.view.default(clamp_max_1, [-1, 1]);  clamp_max_1 = None
        add_4: i32[3456, 5] = torch.ops.aten.add.Tensor(iota_1, view_1)
        sub_2: f32[3456, 5] = torch.ops.aten.sub.Tensor(add_4, view);  add_4 = view = None
        add_5: f32[3456, 5] = torch.ops.aten.add.Tensor(sub_2, 0.5);  sub_2 = None
        mul_1: f32[3456, 5] = torch.ops.aten.mul.Tensor(add_5, 0.7567330851762645);  add_5 = None
        abs_1: f32[3456, 5] = torch.ops.aten.abs.default(mul_1);  mul_1 = None
        clamp_max_2: f32[3456, 5] = torch.ops.aten.clamp_max.default(abs_1, 1.0);  abs_1 = None
        sub_3: f32[3456, 5] = torch.ops.aten.sub.Tensor(1.0, clamp_max_2);  clamp_max_2 = None
        lt: b8[3456, 5] = torch.ops.aten.lt.Tensor(iota_1, view_2);  iota_1 = view_2 = None
        full_default: f32[] = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: f32[3456, 5] = torch.ops.aten.where.self(lt, sub_3, full_default);  lt = sub_3 = full_default = None
        sum_1: f32[3456] = torch.ops.aten.sum.dim_IntList(where, [-1])
        unsqueeze: f32[3456, 1] = torch.ops.aten.unsqueeze.default(sum_1, -1);  sum_1 = None
        div: f32[3456, 5] = torch.ops.aten.div.Tensor(where, unsqueeze);  where = unsqueeze = None
        view_3: i32[3456] = torch.ops.aten.view.default(view_1, [-1]);  view_1 = None
        iota_2: i32[2345] = torch.ops.prims.iota.default(2345, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        add_6: f32[2345] = torch.ops.aten.add.Tensor(iota_2, 0.5);  iota_2 = None
        mul_2: f32[2345] = torch.ops.aten.mul.Tensor(add_6, 1.473773987206823);  add_6 = None
        sub_4: f32[2345] = torch.ops.aten.sub.Tensor(mul_2, 1.473773987206823)
        add_7: f32[2345] = torch.ops.aten.add.Tensor(sub_4, 0.5);  sub_4 = None
        convert_element_type_2: i32[2345] = torch.ops.prims.convert_element_type.default(add_7, torch.int32);  add_7 = None
        clamp_min_1: i32[2345] = torch.ops.aten.clamp_min.default(convert_element_type_2, 0);  convert_element_type_2 = None
        add_8: f32[2345] = torch.ops.aten.add.Tensor(mul_2, 1.473773987206823)
        add_9: f32[2345] = torch.ops.aten.add.Tensor(add_8, 0.5);  add_8 = None
        convert_element_type_3: i32[2345] = torch.ops.prims.convert_element_type.default(add_9, torch.int32);  add_9 = None
        clamp_max_3: i32[2345] = torch.ops.aten.clamp_max.default(convert_element_type_3, 3456);  convert_element_type_3 = None
        sub_5: i32[2345] = torch.ops.aten.sub.Tensor(clamp_max_3, clamp_min_1);  clamp_max_3 = None
        clamp_max_4: i32[2345] = torch.ops.aten.clamp_max.default(sub_5, 5);  sub_5 = None
        iota_3: i32[5] = torch.ops.prims.iota.default(5, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        view_4: f32[2345, 1] = torch.ops.aten.view.default(mul_2, [-1, 1]);  mul_2 = None
        view_5: i32[2345, 1] = torch.ops.aten.view.default(clamp_min_1, [-1, 1]);  clamp_min_1 = None
        view_6: i32[2345, 1] = torch.ops.aten.view.default(clamp_max_4, [-1, 1]);  clamp_max_4 = None
        add_10: i32[2345, 5] = torch.ops.aten.add.Tensor(iota_3, view_5)
        sub_6: f32[2345, 5] = torch.ops.aten.sub.Tensor(add_10, view_4);  add_10 = view_4 = None
        add_11: f32[2345, 5] = torch.ops.aten.add.Tensor(sub_6, 0.5);  sub_6 = None
        mul_3: f32[2345, 5] = torch.ops.aten.mul.Tensor(add_11, 0.6785300925925927);  add_11 = None
        abs_2: f32[2345, 5] = torch.ops.aten.abs.default(mul_3);  mul_3 = None
        clamp_max_5: f32[2345, 5] = torch.ops.aten.clamp_max.default(abs_2, 1.0);  abs_2 = None
        sub_7: f32[2345, 5] = torch.ops.aten.sub.Tensor(1.0, clamp_max_5);  clamp_max_5 = None
        lt_1: b8[2345, 5] = torch.ops.aten.lt.Tensor(iota_3, view_6);  iota_3 = view_6 = None
        full_default_1: f32[] = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: f32[2345, 5] = torch.ops.aten.where.self(lt_1, sub_7, full_default_1);  lt_1 = sub_7 = full_default_1 = None
        sum_2: f32[2345] = torch.ops.aten.sum.dim_IntList(where_1, [-1])
        unsqueeze_1: f32[2345, 1] = torch.ops.aten.unsqueeze.default(sum_2, -1);  sum_2 = None
        div_1: f32[2345, 5] = torch.ops.aten.div.Tensor(where_1, unsqueeze_1);  where_1 = unsqueeze_1 = None
        view_7: i32[2345] = torch.ops.aten.view.default(view_5, [-1]);  view_5 = None
        iota_4: i64[5] = torch.ops.prims.iota.default(5, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        iota_5: i64[5] = torch.ops.prims.iota.default(5, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_2: i32[3456, 1] = torch.ops.aten.unsqueeze.default(view_3, -1);  view_3 = None
        unsqueeze_3: i32[2345, 1] = torch.ops.aten.unsqueeze.default(view_7, -1);  view_7 = None
        add_12: i64[3456, 5] = torch.ops.aten.add.Tensor(unsqueeze_2, iota_4);  unsqueeze_2 = iota_4 = None
        clamp_max_6: i64[3456, 5] = torch.ops.aten.clamp_max.default(add_12, 4566);  add_12 = None
        add_13: i64[2345, 5] = torch.ops.aten.add.Tensor(unsqueeze_3, iota_5);  unsqueeze_3 = iota_5 = None
        clamp_max_7: i64[2345, 5] = torch.ops.aten.clamp_max.default(add_13, 3455);  add_13 = None
        view_8: i64[2345, 5, 1, 1] = torch.ops.aten.view.default(clamp_max_7, [2345, 5, 1, 1]);  clamp_max_7 = None
        slice_1: f32[1, 3, 3456, 4567] = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, 9223372036854775807);  arg0_1 = None
        slice_2: f32[1, 3, 3456, 4567] = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 9223372036854775807);  slice_1 = None
        index: f32[1, 3, 2345, 5, 3456, 5] = torch.ops.aten.index.Tensor(slice_2, [None, None, view_8, clamp_max_6]);  slice_2 = view_8 = clamp_max_6 = None
        view_9: f32[2345, 5, 1, 1] = torch.ops.aten.view.default(div_1, [2345, 5, 1, 1]);  div_1 = None
        view_10: f32[1, 3456, 5] = torch.ops.aten.view.default(div, [1, 3456, 5]);  div = None
        mul_4: f32[1, 3, 2345, 5, 3456, 5] = torch.ops.aten.mul.Tensor(view_10, index);  view_10 = index = None
        mul_5: f32[1, 3, 2345, 5, 3456, 5] = torch.ops.aten.mul.Tensor(view_9, mul_4);  view_9 = mul_4 = None
        sum_3: f32[1, 3, 2345, 3456] = torch.ops.aten.sum.dim_IntList(mul_5, [-1, -3]);  mul_5 = None
        return (sum_3,)
        