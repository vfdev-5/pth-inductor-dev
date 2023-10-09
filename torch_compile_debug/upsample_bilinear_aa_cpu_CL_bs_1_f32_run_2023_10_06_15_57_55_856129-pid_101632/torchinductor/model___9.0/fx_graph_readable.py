class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: f32[1, 3, 345, 456]):
        # File: check_interpolate_bilinear_aa.py:12, code: img = torch.nn.functional.interpolate(img, size=(271, 272), mode="bilinear", antialias=True)
        iota: i64[1] = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        full_default: i64[1, 1, 1, 1] = torch.ops.aten.full.default([1, 1, 1, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        iota_1: i64[3] = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_1: i64[1, 3, 1, 1] = torch.ops.aten.view.default(iota_1, [1, 3, 1, 1]);  iota_1 = None
        iota_2: i64[272] = torch.ops.prims.iota.default(272, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        add: f32[272] = torch.ops.aten.add.Tensor(iota_2, 0.5);  iota_2 = None
        mul: f32[272] = torch.ops.aten.mul.Tensor(add, 1.6764705882352942);  add = None
        sub: f32[272] = torch.ops.aten.sub.Tensor(mul, 1.6764705882352942)
        add_1: f32[272] = torch.ops.aten.add.Tensor(sub, 0.5);  sub = None
        convert_element_type: i64[272] = torch.ops.prims.convert_element_type.default(add_1, torch.int64);  add_1 = None
        clamp_min: i64[272] = torch.ops.aten.clamp_min.default(convert_element_type, 0);  convert_element_type = None
        add_2: f32[272] = torch.ops.aten.add.Tensor(mul, 1.6764705882352942)
        add_3: f32[272] = torch.ops.aten.add.Tensor(add_2, 0.5);  add_2 = None
        convert_element_type_1: i64[272] = torch.ops.prims.convert_element_type.default(add_3, torch.int64);  add_3 = None
        clamp_max: i64[272] = torch.ops.aten.clamp_max.default(convert_element_type_1, 456);  convert_element_type_1 = None
        sub_1: i64[272] = torch.ops.aten.sub.Tensor(clamp_max, clamp_min);  clamp_max = None
        clamp_max_1: i64[272] = torch.ops.aten.clamp_max.default(sub_1, 5);  sub_1 = None
        iota_3: i64[5] = torch.ops.prims.iota.default(5, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_2: i64[5, 1] = torch.ops.aten.view.default(iota_3, [-1, 1]);  iota_3 = None
        add_4: i64[5, 272] = torch.ops.aten.add.Tensor(view_2, clamp_min)
        sub_2: f32[5, 272] = torch.ops.aten.sub.Tensor(add_4, mul);  add_4 = mul = None
        add_5: f32[5, 272] = torch.ops.aten.add.Tensor(sub_2, 0.5);  sub_2 = None
        mul_1: f32[5, 272] = torch.ops.aten.mul.Tensor(add_5, 0.5964912280701754);  add_5 = None
        abs_1: f32[5, 272] = torch.ops.aten.abs.default(mul_1);  mul_1 = None
        clamp_max_2: f32[5, 272] = torch.ops.aten.clamp_max.default(abs_1, 1.0);  abs_1 = None
        sub_3: f32[5, 272] = torch.ops.aten.sub.Tensor(1.0, clamp_max_2);  clamp_max_2 = None
        lt: b8[5, 272] = torch.ops.aten.lt.Tensor(view_2, clamp_max_1);  view_2 = clamp_max_1 = None
        full_default_1: f32[] = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where: f32[5, 272] = torch.ops.aten.where.self(lt, sub_3, full_default_1);  lt = sub_3 = full_default_1 = None
        sum_1: f32[272] = torch.ops.aten.sum.dim_IntList(where, [0])
        div: f32[5, 272] = torch.ops.aten.div.Tensor(where, sum_1);  where = sum_1 = None
        iota_4: i64[345] = torch.ops.prims.iota.default(345, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_3: i64[1, 1, 345, 1] = torch.ops.aten.view.default(iota_4, [1, 1, 345, 1]);  iota_4 = None
        view_4: i64[1, 1, 1, 272] = torch.ops.aten.view.default(clamp_min, [1, 1, 1, 272]);  clamp_min = None
        add_6: i64[1, 1, 1, 272] = torch.ops.aten.add.Tensor(view_4, 0)
        clamp_max_3: i64[1, 1, 1, 272] = torch.ops.aten.clamp_max.default(add_6, 455);  add_6 = None
        index: f32[1, 3, 345, 272] = torch.ops.aten.index.Tensor(arg0_1, [full_default, view_1, view_3, clamp_max_3]);  clamp_max_3 = None
        clone: f32[1, 3, 345, 272] = torch.ops.aten.clone.default(index, memory_format = torch.channels_last);  index = None
        add_7: i64[1, 1, 1, 272] = torch.ops.aten.add.Tensor(view_4, 1)
        clamp_max_4: i64[1, 1, 1, 272] = torch.ops.aten.clamp_max.default(add_7, 455);  add_7 = None
        index_1: f32[1, 3, 345, 272] = torch.ops.aten.index.Tensor(arg0_1, [full_default, view_1, view_3, clamp_max_4]);  clamp_max_4 = None
        clone_1: f32[1, 3, 345, 272] = torch.ops.aten.clone.default(index_1, memory_format = torch.channels_last);  index_1 = None
        add_8: i64[1, 1, 1, 272] = torch.ops.aten.add.Tensor(view_4, 2)
        clamp_max_5: i64[1, 1, 1, 272] = torch.ops.aten.clamp_max.default(add_8, 455);  add_8 = None
        index_2: f32[1, 3, 345, 272] = torch.ops.aten.index.Tensor(arg0_1, [full_default, view_1, view_3, clamp_max_5]);  clamp_max_5 = None
        clone_2: f32[1, 3, 345, 272] = torch.ops.aten.clone.default(index_2, memory_format = torch.channels_last);  index_2 = None
        add_9: i64[1, 1, 1, 272] = torch.ops.aten.add.Tensor(view_4, 3)
        clamp_max_6: i64[1, 1, 1, 272] = torch.ops.aten.clamp_max.default(add_9, 455);  add_9 = None
        index_3: f32[1, 3, 345, 272] = torch.ops.aten.index.Tensor(arg0_1, [full_default, view_1, view_3, clamp_max_6]);  clamp_max_6 = None
        clone_3: f32[1, 3, 345, 272] = torch.ops.aten.clone.default(index_3, memory_format = torch.channels_last);  index_3 = None
        add_10: i64[1, 1, 1, 272] = torch.ops.aten.add.Tensor(view_4, 4);  view_4 = None
        clamp_max_7: i64[1, 1, 1, 272] = torch.ops.aten.clamp_max.default(add_10, 455);  add_10 = None
        index_4: f32[1, 3, 345, 272] = torch.ops.aten.index.Tensor(arg0_1, [full_default, view_1, view_3, clamp_max_7]);  arg0_1 = full_default = view_1 = view_3 = clamp_max_7 = None
        clone_4: f32[1, 3, 345, 272] = torch.ops.aten.clone.default(index_4, memory_format = torch.channels_last);  index_4 = None
        unbind = torch.ops.aten.unbind.int(div);  div = None
        getitem: f32[272] = unbind[0]
        getitem_1: f32[272] = unbind[1]
        getitem_2: f32[272] = unbind[2]
        getitem_3: f32[272] = unbind[3];  unbind = None
        full_default_2: f32[272] = torch.ops.aten.full.default([272], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        mul_2: f32[1, 3, 345, 272] = torch.ops.aten.mul.Tensor(clone, getitem);  clone = getitem = None
        mul_3: f32[1, 3, 345, 272] = torch.ops.aten.mul.Tensor(clone_1, getitem_1);  clone_1 = getitem_1 = None
        add_11: f32[1, 3, 345, 272] = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        mul_4: f32[1, 3, 345, 272] = torch.ops.aten.mul.Tensor(clone_2, getitem_2);  clone_2 = getitem_2 = None
        add_12: f32[1, 3, 345, 272] = torch.ops.aten.add.Tensor(add_11, mul_4);  add_11 = mul_4 = None
        mul_5: f32[1, 3, 345, 272] = torch.ops.aten.mul.Tensor(clone_3, getitem_3);  clone_3 = getitem_3 = None
        add_13: f32[1, 3, 345, 272] = torch.ops.aten.add.Tensor(add_12, mul_5);  add_12 = mul_5 = None
        mul_6: f32[1, 3, 345, 272] = torch.ops.aten.mul.Tensor(clone_4, full_default_2);  clone_4 = full_default_2 = None
        add_14: f32[1, 3, 345, 272] = torch.ops.aten.add.Tensor(add_13, mul_6);  add_13 = mul_6 = None
        iota_5: i64[1] = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        full_default_3: i64[1, 1, 1, 1] = torch.ops.aten.full.default([1, 1, 1, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        iota_6: i64[3] = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_6: i64[1, 3, 1, 1] = torch.ops.aten.view.default(iota_6, [1, 3, 1, 1]);  iota_6 = None
        iota_7: i64[271] = torch.ops.prims.iota.default(271, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        add_15: f32[271] = torch.ops.aten.add.Tensor(iota_7, 0.5);  iota_7 = None
        mul_7: f32[271] = torch.ops.aten.mul.Tensor(add_15, 1.2730627306273063);  add_15 = None
        sub_4: f32[271] = torch.ops.aten.sub.Tensor(mul_7, 1.2730627306273063)
        add_16: f32[271] = torch.ops.aten.add.Tensor(sub_4, 0.5);  sub_4 = None
        convert_element_type_2: i64[271] = torch.ops.prims.convert_element_type.default(add_16, torch.int64);  add_16 = None
        clamp_min_1: i64[271] = torch.ops.aten.clamp_min.default(convert_element_type_2, 0);  convert_element_type_2 = None
        add_17: f32[271] = torch.ops.aten.add.Tensor(mul_7, 1.2730627306273063)
        add_18: f32[271] = torch.ops.aten.add.Tensor(add_17, 0.5);  add_17 = None
        convert_element_type_3: i64[271] = torch.ops.prims.convert_element_type.default(add_18, torch.int64);  add_18 = None
        clamp_max_8: i64[271] = torch.ops.aten.clamp_max.default(convert_element_type_3, 345);  convert_element_type_3 = None
        sub_5: i64[271] = torch.ops.aten.sub.Tensor(clamp_max_8, clamp_min_1);  clamp_max_8 = None
        clamp_max_9: i64[271] = torch.ops.aten.clamp_max.default(sub_5, 5);  sub_5 = None
        iota_8: i64[5] = torch.ops.prims.iota.default(5, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_7: i64[5, 1] = torch.ops.aten.view.default(iota_8, [-1, 1]);  iota_8 = None
        add_19: i64[5, 271] = torch.ops.aten.add.Tensor(view_7, clamp_min_1)
        sub_6: f32[5, 271] = torch.ops.aten.sub.Tensor(add_19, mul_7);  add_19 = mul_7 = None
        add_20: f32[5, 271] = torch.ops.aten.add.Tensor(sub_6, 0.5);  sub_6 = None
        mul_8: f32[5, 271] = torch.ops.aten.mul.Tensor(add_20, 0.7855072463768116);  add_20 = None
        abs_2: f32[5, 271] = torch.ops.aten.abs.default(mul_8);  mul_8 = None
        clamp_max_10: f32[5, 271] = torch.ops.aten.clamp_max.default(abs_2, 1.0);  abs_2 = None
        sub_7: f32[5, 271] = torch.ops.aten.sub.Tensor(1.0, clamp_max_10);  clamp_max_10 = None
        lt_1: b8[5, 271] = torch.ops.aten.lt.Tensor(view_7, clamp_max_9);  view_7 = clamp_max_9 = None
        full_default_4: f32[] = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_1: f32[5, 271] = torch.ops.aten.where.self(lt_1, sub_7, full_default_4);  lt_1 = sub_7 = full_default_4 = None
        sum_2: f32[271] = torch.ops.aten.sum.dim_IntList(where_1, [0])
        div_1: f32[5, 271] = torch.ops.aten.div.Tensor(where_1, sum_2);  where_1 = sum_2 = None
        iota_9: i64[272] = torch.ops.prims.iota.default(272, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_8: i64[1, 1, 1, 272] = torch.ops.aten.view.default(iota_9, [1, 1, 1, 272]);  iota_9 = None
        view_9: i64[1, 1, 271, 1] = torch.ops.aten.view.default(clamp_min_1, [1, 1, 271, 1]);  clamp_min_1 = None
        add_21: i64[1, 1, 271, 1] = torch.ops.aten.add.Tensor(view_9, 0)
        clamp_max_11: i64[1, 1, 271, 1] = torch.ops.aten.clamp_max.default(add_21, 344);  add_21 = None
        index_5: f32[1, 3, 271, 272] = torch.ops.aten.index.Tensor(add_14, [full_default_3, view_6, clamp_max_11, view_8]);  clamp_max_11 = None
        clone_5: f32[1, 3, 271, 272] = torch.ops.aten.clone.default(index_5, memory_format = torch.channels_last);  index_5 = None
        add_22: i64[1, 1, 271, 1] = torch.ops.aten.add.Tensor(view_9, 1)
        clamp_max_12: i64[1, 1, 271, 1] = torch.ops.aten.clamp_max.default(add_22, 344);  add_22 = None
        index_6: f32[1, 3, 271, 272] = torch.ops.aten.index.Tensor(add_14, [full_default_3, view_6, clamp_max_12, view_8]);  clamp_max_12 = None
        clone_6: f32[1, 3, 271, 272] = torch.ops.aten.clone.default(index_6, memory_format = torch.channels_last);  index_6 = None
        add_23: i64[1, 1, 271, 1] = torch.ops.aten.add.Tensor(view_9, 2)
        clamp_max_13: i64[1, 1, 271, 1] = torch.ops.aten.clamp_max.default(add_23, 344);  add_23 = None
        index_7: f32[1, 3, 271, 272] = torch.ops.aten.index.Tensor(add_14, [full_default_3, view_6, clamp_max_13, view_8]);  clamp_max_13 = None
        clone_7: f32[1, 3, 271, 272] = torch.ops.aten.clone.default(index_7, memory_format = torch.channels_last);  index_7 = None
        add_24: i64[1, 1, 271, 1] = torch.ops.aten.add.Tensor(view_9, 3)
        clamp_max_14: i64[1, 1, 271, 1] = torch.ops.aten.clamp_max.default(add_24, 344);  add_24 = None
        index_8: f32[1, 3, 271, 272] = torch.ops.aten.index.Tensor(add_14, [full_default_3, view_6, clamp_max_14, view_8]);  clamp_max_14 = None
        clone_8: f32[1, 3, 271, 272] = torch.ops.aten.clone.default(index_8, memory_format = torch.channels_last);  index_8 = None
        add_25: i64[1, 1, 271, 1] = torch.ops.aten.add.Tensor(view_9, 4);  view_9 = None
        clamp_max_15: i64[1, 1, 271, 1] = torch.ops.aten.clamp_max.default(add_25, 344);  add_25 = None
        index_9: f32[1, 3, 271, 272] = torch.ops.aten.index.Tensor(add_14, [full_default_3, view_6, clamp_max_15, view_8]);  add_14 = full_default_3 = view_6 = clamp_max_15 = view_8 = None
        clone_9: f32[1, 3, 271, 272] = torch.ops.aten.clone.default(index_9, memory_format = torch.channels_last);  index_9 = None
        unsqueeze: f32[5, 271, 1] = torch.ops.aten.unsqueeze.default(div_1, -1);  div_1 = None
        unbind_1 = torch.ops.aten.unbind.int(unsqueeze);  unsqueeze = None
        getitem_5: f32[271, 1] = unbind_1[0]
        getitem_6: f32[271, 1] = unbind_1[1]
        getitem_7: f32[271, 1] = unbind_1[2]
        getitem_8: f32[271, 1] = unbind_1[3]
        getitem_9: f32[271, 1] = unbind_1[4];  unbind_1 = None
        mul_9: f32[1, 3, 271, 272] = torch.ops.aten.mul.Tensor(clone_5, getitem_5);  clone_5 = getitem_5 = None
        mul_10: f32[1, 3, 271, 272] = torch.ops.aten.mul.Tensor(clone_6, getitem_6);  clone_6 = getitem_6 = None
        add_26: f32[1, 3, 271, 272] = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
        mul_11: f32[1, 3, 271, 272] = torch.ops.aten.mul.Tensor(clone_7, getitem_7);  clone_7 = getitem_7 = None
        add_27: f32[1, 3, 271, 272] = torch.ops.aten.add.Tensor(add_26, mul_11);  add_26 = mul_11 = None
        mul_12: f32[1, 3, 271, 272] = torch.ops.aten.mul.Tensor(clone_8, getitem_8);  clone_8 = getitem_8 = None
        add_28: f32[1, 3, 271, 272] = torch.ops.aten.add.Tensor(add_27, mul_12);  add_27 = mul_12 = None
        mul_13: f32[1, 3, 271, 272] = torch.ops.aten.mul.Tensor(clone_9, getitem_9);  clone_9 = getitem_9 = None
        add_29: f32[1, 3, 271, 272] = torch.ops.aten.add.Tensor(add_28, mul_13);  add_28 = mul_13 = None
        return (add_29,)
        