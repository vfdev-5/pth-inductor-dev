class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: Sym(s0), arg1_1: f32[s0, 3, 345, 456], arg2_1: f32[s0, 2, 3]):
        # File: check_affine_grid_sampler.py:24, code: grid = affine_grid(theta, size=(n, c, h, w), align_corners=align_corners)
        iota: i64[456] = torch.ops.prims.iota.default(456, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        lt: b8[456] = torch.ops.aten.lt.Scalar(iota, 228.0)
        convert_element_type: f32[456] = torch.ops.prims.convert_element_type.default(iota, torch.float32)
        mul: f32[456] = torch.ops.aten.mul.Tensor(convert_element_type, 0.004385964912280702);  convert_element_type = None
        add: f32[456] = torch.ops.aten.add.Tensor(mul, -0.9978070175438597);  mul = None
        sub: i64[456] = torch.ops.aten.sub.Tensor(455, iota);  iota = None
        convert_element_type_1: f32[456] = torch.ops.prims.convert_element_type.default(sub, torch.float32);  sub = None
        mul_1: f32[456] = torch.ops.aten.mul.Tensor(convert_element_type_1, 0.004385964912280702);  convert_element_type_1 = None
        sub_1: f32[456] = torch.ops.aten.sub.Tensor(0.9978070175438597, mul_1);  mul_1 = None
        where: f32[456] = torch.ops.aten.where.self(lt, add, sub_1);  lt = add = sub_1 = None
        view: f32[1, 456, 1] = torch.ops.aten.view.default(where, [1, 456, 1]);  where = None
        iota_1: i64[345] = torch.ops.prims.iota.default(345, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        lt_1: b8[345] = torch.ops.aten.lt.Scalar(iota_1, 172.5)
        convert_element_type_2: f32[345] = torch.ops.prims.convert_element_type.default(iota_1, torch.float32)
        mul_2: f32[345] = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.005797101449275362);  convert_element_type_2 = None
        add_1: f32[345] = torch.ops.aten.add.Tensor(mul_2, -0.9971014492753624);  mul_2 = None
        sub_2: i64[345] = torch.ops.aten.sub.Tensor(344, iota_1);  iota_1 = None
        convert_element_type_3: f32[345] = torch.ops.prims.convert_element_type.default(sub_2, torch.float32);  sub_2 = None
        mul_3: f32[345] = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.005797101449275362);  convert_element_type_3 = None
        sub_3: f32[345] = torch.ops.aten.sub.Tensor(0.9971014492753624, mul_3);  mul_3 = None
        where_1: f32[345] = torch.ops.aten.where.self(lt_1, add_1, sub_3);  lt_1 = add_1 = sub_3 = None
        view_1: f32[345, 1, 1] = torch.ops.aten.view.default(where_1, [345, 1, 1]);  where_1 = None
        full_default: f32[1, 1, 1] = torch.ops.aten.full.default([1, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        constant_pad_nd: f32[1, 456, 3] = torch.ops.aten.constant_pad_nd.default(view, [0, 2], 0.0);  view = None
        constant_pad_nd_1: f32[345, 1, 3] = torch.ops.aten.constant_pad_nd.default(view_1, [1, 1], 0.0);  view_1 = None
        constant_pad_nd_2: f32[1, 1, 3] = torch.ops.aten.constant_pad_nd.default(full_default, [2, 0], 0.0);  full_default = None
        add_2: f32[345, 456, 3] = torch.ops.aten.add.Tensor(constant_pad_nd, constant_pad_nd_1);  constant_pad_nd = constant_pad_nd_1 = None
        add_3: f32[345, 456, 3] = torch.ops.aten.add.Tensor(add_2, constant_pad_nd_2);  add_2 = constant_pad_nd_2 = None
        view_2: f32[157320, 3, 1] = torch.ops.aten.view.default(add_3, [-1, 3, 1]);  add_3 = None
        permute: f32[s0, 3, 2] = torch.ops.aten.permute.default(arg2_1, [0, 2, 1]);  arg2_1 = None
        unsqueeze: f32[s0, 1, 3, 2] = torch.ops.aten.unsqueeze.default(permute, 1);  permute = None
        mul_4: f32[s0, 157320, 3, 2] = torch.ops.aten.mul.Tensor(view_2, unsqueeze);  view_2 = unsqueeze = None
        sum_1: f32[s0, 157320, 2] = torch.ops.aten.sum.dim_IntList(mul_4, [-2]);  mul_4 = None
        view_3: f32[s0, 345, 456, 2] = torch.ops.aten.view.default(sum_1, [arg0_1, 345, 456, 2]);  sum_1 = None
        
        # File: check_affine_grid_sampler.py:25, code: output = grid_sample(img, grid, align_corners=align_corners, mode=mode)
        view_4: f32[s0, 1, 345, 456, 2] = torch.ops.aten.view.default(view_3, [arg0_1, 1, 345, 456, 2]);  view_3 = None
        expand: f32[s0, 3, 345, 456, 2] = torch.ops.aten.expand.default(view_4, [arg0_1, 3, 345, 456, 2]);  view_4 = None
        sub_4: Sym(s0) = arg0_1 - 0
        add_4: Sym(s0 + 1) = sub_4 + 1;  sub_4 = None
        sub_5: Sym(s0) = add_4 - 1;  add_4 = None
        floordiv: Sym(s0) = sub_5 // 1;  sub_5 = None
        iota_2: i64[s0] = torch.ops.prims.iota.default(floordiv, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False);  floordiv = None
        view_5: i64[s0, 1, 1, 1] = torch.ops.aten.view.default(iota_2, [arg0_1, 1, 1, 1]);  iota_2 = None
        iota_3: i64[3] = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_6: i64[1, 3, 1, 1] = torch.ops.aten.view.default(iota_3, [1, 3, 1, 1]);  iota_3 = None
        select: f32[s0, 3, 345, 456] = torch.ops.aten.select.int(expand, 4, 0)
        select_1: f32[s0, 3, 345, 456] = torch.ops.aten.select.int(expand, 4, 1);  expand = None
        mul_5: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(select, 228.0);  select = None
        add_5: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_5, 227.5);  mul_5 = None
        mul_6: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(select_1, 172.5);  select_1 = None
        add_6: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_6, 172.0);  mul_6 = None
        floor: f32[s0, 3, 345, 456] = torch.ops.aten.floor.default(add_5)
        floor_1: f32[s0, 3, 345, 456] = torch.ops.aten.floor.default(add_6)
        sub_6: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(add_5, floor);  add_5 = None
        sub_7: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(add_6, floor_1);  add_6 = None
        add_7: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(floor_1, -1)
        sub_8: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(floor, 1)
        ge: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(sub_8, 0)
        lt_2: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(sub_8, 456)
        ge_1: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_7, 0)
        lt_3: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_7, 345)
        logical_and: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_1, lt_3);  ge_1 = lt_3 = None
        logical_and_1: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(lt_2, logical_and);  lt_2 = logical_and = None
        logical_and_2: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge, logical_and_1);  ge = logical_and_1 = None
        convert_element_type_4: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(sub_8, torch.int64);  sub_8 = None
        convert_element_type_5: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_7, torch.int64)
        full_default_1: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_2, convert_element_type_4, full_default_1);  convert_element_type_4 = full_default_1 = None
        view_7: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_2, [arg0_1, 3, 345, 456]);  where_2 = None
        full_default_2: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_2, convert_element_type_5, full_default_2);  convert_element_type_5 = full_default_2 = None
        view_8: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_3, [arg0_1, 3, 345, 456]);  where_3 = None
        full_default_3: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: i64[] = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_2, full_default_4, full_default_3);  logical_and_2 = full_default_4 = full_default_3 = None
        view_9: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_4, [arg0_1, 3, 345, 456]);  where_4 = None
        index: f32[s0, 3, 345, 456] = torch.ops.aten.index.Tensor(arg1_1, [view_5, view_6, view_8, view_7]);  view_8 = view_7 = None
        mul_7: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(index, view_9);  index = view_9 = None
        ge_2: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(floor, 0)
        lt_4: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(floor, 456)
        ge_3: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_7, 0)
        lt_5: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_7, 345)
        logical_and_3: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_3, lt_5);  ge_3 = lt_5 = None
        logical_and_4: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(lt_4, logical_and_3);  lt_4 = logical_and_3 = None
        logical_and_5: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_2, logical_and_4);  ge_2 = logical_and_4 = None
        convert_element_type_6: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(floor, torch.int64)
        convert_element_type_7: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_7, torch.int64)
        full_default_5: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_5, convert_element_type_6, full_default_5);  convert_element_type_6 = full_default_5 = None
        view_10: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_5, [arg0_1, 3, 345, 456]);  where_5 = None
        full_default_6: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_5, convert_element_type_7, full_default_6);  convert_element_type_7 = full_default_6 = None
        view_11: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_6, [arg0_1, 3, 345, 456]);  where_6 = None
        full_default_7: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: i64[] = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_5, full_default_8, full_default_7);  logical_and_5 = full_default_8 = full_default_7 = None
        view_12: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_7, [arg0_1, 3, 345, 456]);  where_7 = None
        index_1: f32[s0, 3, 345, 456] = torch.ops.aten.index.Tensor(arg1_1, [view_5, view_6, view_11, view_10]);  view_11 = view_10 = None
        mul_8: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(index_1, view_12);  index_1 = view_12 = None
        add_8: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(floor, 1)
        ge_4: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_8, 0)
        lt_6: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_8, 456)
        ge_5: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_7, 0)
        lt_7: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_7, 345)
        logical_and_6: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_5, lt_7);  ge_5 = lt_7 = None
        logical_and_7: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(lt_6, logical_and_6);  lt_6 = logical_and_6 = None
        logical_and_8: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_4, logical_and_7);  ge_4 = logical_and_7 = None
        convert_element_type_8: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_8, torch.int64);  add_8 = None
        convert_element_type_9: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_7, torch.int64)
        full_default_9: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_8, convert_element_type_8, full_default_9);  convert_element_type_8 = full_default_9 = None
        view_13: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_8, [arg0_1, 3, 345, 456]);  where_8 = None
        full_default_10: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_8, convert_element_type_9, full_default_10);  convert_element_type_9 = full_default_10 = None
        view_14: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_9, [arg0_1, 3, 345, 456]);  where_9 = None
        full_default_11: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: i64[] = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_8, full_default_12, full_default_11);  logical_and_8 = full_default_12 = full_default_11 = None
        view_15: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_10, [arg0_1, 3, 345, 456]);  where_10 = None
        index_2: f32[s0, 3, 345, 456] = torch.ops.aten.index.Tensor(arg1_1, [view_5, view_6, view_14, view_13]);  view_14 = view_13 = None
        mul_9: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(index_2, view_15);  index_2 = view_15 = None
        add_9: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(floor, 2)
        ge_6: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_9, 0)
        lt_8: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_9, 456)
        ge_7: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_7, 0)
        lt_9: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_7, 345)
        logical_and_9: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_7, lt_9);  ge_7 = lt_9 = None
        logical_and_10: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(lt_8, logical_and_9);  lt_8 = logical_and_9 = None
        logical_and_11: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_6, logical_and_10);  ge_6 = logical_and_10 = None
        convert_element_type_10: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_9, torch.int64);  add_9 = None
        convert_element_type_11: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_7, torch.int64);  add_7 = None
        full_default_13: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_11, convert_element_type_10, full_default_13);  convert_element_type_10 = full_default_13 = None
        view_16: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_11, [arg0_1, 3, 345, 456]);  where_11 = None
        full_default_14: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_11, convert_element_type_11, full_default_14);  convert_element_type_11 = full_default_14 = None
        view_17: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_12, [arg0_1, 3, 345, 456]);  where_12 = None
        full_default_15: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: i64[] = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_11, full_default_16, full_default_15);  logical_and_11 = full_default_16 = full_default_15 = None
        view_18: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_13, [arg0_1, 3, 345, 456]);  where_13 = None
        index_3: f32[s0, 3, 345, 456] = torch.ops.aten.index.Tensor(arg1_1, [view_5, view_6, view_17, view_16]);  view_17 = view_16 = None
        mul_10: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(index_3, view_18);  index_3 = view_18 = None
        add_10: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(sub_6, 1.0)
        mul_11: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(add_10, -0.75)
        sub_9: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_11, -3.75);  mul_11 = None
        mul_12: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_9, add_10);  sub_9 = None
        add_11: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_12, -6.0);  mul_12 = None
        mul_13: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(add_11, add_10);  add_11 = add_10 = None
        sub_10: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_13, -3.0);  mul_13 = None
        mul_14: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_6, 1.25)
        sub_11: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_14, 2.25);  mul_14 = None
        mul_15: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_11, sub_6);  sub_11 = None
        mul_16: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_15, sub_6);  mul_15 = None
        add_12: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_16, 1);  mul_16 = None
        sub_12: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(1.0, sub_6)
        mul_17: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_12, 1.25)
        sub_13: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_17, 2.25);  mul_17 = None
        mul_18: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_13, sub_12);  sub_13 = None
        mul_19: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_18, sub_12);  mul_18 = sub_12 = None
        add_13: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_19, 1);  mul_19 = None
        sub_14: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(2.0, sub_6)
        mul_20: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_14, -0.75)
        sub_15: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_20, -3.75);  mul_20 = None
        mul_21: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_15, sub_14);  sub_15 = None
        add_14: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_21, -6.0);  mul_21 = None
        mul_22: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(add_14, sub_14);  add_14 = sub_14 = None
        sub_16: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_22, -3.0);  mul_22 = None
        mul_23: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_7, sub_10);  mul_7 = sub_10 = None
        mul_24: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_8, add_12);  mul_8 = add_12 = None
        add_15: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_23, mul_24);  mul_23 = mul_24 = None
        mul_25: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_9, add_13);  mul_9 = add_13 = None
        add_16: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(add_15, mul_25);  add_15 = mul_25 = None
        mul_26: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_10, sub_16);  mul_10 = sub_16 = None
        add_17: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(add_16, mul_26);  add_16 = mul_26 = None
        add_18: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(floor_1, 0)
        sub_17: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(floor, 1)
        ge_8: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(sub_17, 0)
        lt_10: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(sub_17, 456)
        ge_9: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_18, 0)
        lt_11: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_18, 345)
        logical_and_12: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_9, lt_11);  ge_9 = lt_11 = None
        logical_and_13: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(lt_10, logical_and_12);  lt_10 = logical_and_12 = None
        logical_and_14: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_8, logical_and_13);  ge_8 = logical_and_13 = None
        convert_element_type_12: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(sub_17, torch.int64);  sub_17 = None
        convert_element_type_13: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_18, torch.int64)
        full_default_17: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_14: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_14, convert_element_type_12, full_default_17);  convert_element_type_12 = full_default_17 = None
        view_19: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_14, [arg0_1, 3, 345, 456]);  where_14 = None
        full_default_18: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_15: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_14, convert_element_type_13, full_default_18);  convert_element_type_13 = full_default_18 = None
        view_20: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_15, [arg0_1, 3, 345, 456]);  where_15 = None
        full_default_19: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20: i64[] = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_14, full_default_20, full_default_19);  logical_and_14 = full_default_20 = full_default_19 = None
        view_21: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_16, [arg0_1, 3, 345, 456]);  where_16 = None
        index_4: f32[s0, 3, 345, 456] = torch.ops.aten.index.Tensor(arg1_1, [view_5, view_6, view_20, view_19]);  view_20 = view_19 = None
        mul_27: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(index_4, view_21);  index_4 = view_21 = None
        ge_10: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(floor, 0)
        lt_12: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(floor, 456)
        ge_11: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_18, 0)
        lt_13: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_18, 345)
        logical_and_15: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_11, lt_13);  ge_11 = lt_13 = None
        logical_and_16: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(lt_12, logical_and_15);  lt_12 = logical_and_15 = None
        logical_and_17: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_10, logical_and_16);  ge_10 = logical_and_16 = None
        convert_element_type_14: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(floor, torch.int64)
        convert_element_type_15: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_18, torch.int64)
        full_default_21: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_17: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_17, convert_element_type_14, full_default_21);  convert_element_type_14 = full_default_21 = None
        view_22: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_17, [arg0_1, 3, 345, 456]);  where_17 = None
        full_default_22: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_18: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_17, convert_element_type_15, full_default_22);  convert_element_type_15 = full_default_22 = None
        view_23: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_18, [arg0_1, 3, 345, 456]);  where_18 = None
        full_default_23: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24: i64[] = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_19: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_17, full_default_24, full_default_23);  logical_and_17 = full_default_24 = full_default_23 = None
        view_24: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_19, [arg0_1, 3, 345, 456]);  where_19 = None
        index_5: f32[s0, 3, 345, 456] = torch.ops.aten.index.Tensor(arg1_1, [view_5, view_6, view_23, view_22]);  view_23 = view_22 = None
        mul_28: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(index_5, view_24);  index_5 = view_24 = None
        add_19: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(floor, 1)
        ge_12: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_19, 0)
        lt_14: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_19, 456)
        ge_13: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_18, 0)
        lt_15: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_18, 345)
        logical_and_18: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_13, lt_15);  ge_13 = lt_15 = None
        logical_and_19: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(lt_14, logical_and_18);  lt_14 = logical_and_18 = None
        logical_and_20: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_12, logical_and_19);  ge_12 = logical_and_19 = None
        convert_element_type_16: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_19, torch.int64);  add_19 = None
        convert_element_type_17: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_18, torch.int64)
        full_default_25: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_20: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_20, convert_element_type_16, full_default_25);  convert_element_type_16 = full_default_25 = None
        view_25: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_20, [arg0_1, 3, 345, 456]);  where_20 = None
        full_default_26: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_21: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_20, convert_element_type_17, full_default_26);  convert_element_type_17 = full_default_26 = None
        view_26: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_21, [arg0_1, 3, 345, 456]);  where_21 = None
        full_default_27: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_28: i64[] = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_22: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_20, full_default_28, full_default_27);  logical_and_20 = full_default_28 = full_default_27 = None
        view_27: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_22, [arg0_1, 3, 345, 456]);  where_22 = None
        index_6: f32[s0, 3, 345, 456] = torch.ops.aten.index.Tensor(arg1_1, [view_5, view_6, view_26, view_25]);  view_26 = view_25 = None
        mul_29: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(index_6, view_27);  index_6 = view_27 = None
        add_20: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(floor, 2)
        ge_14: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_20, 0)
        lt_16: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_20, 456)
        ge_15: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_18, 0)
        lt_17: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_18, 345)
        logical_and_21: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_15, lt_17);  ge_15 = lt_17 = None
        logical_and_22: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(lt_16, logical_and_21);  lt_16 = logical_and_21 = None
        logical_and_23: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_14, logical_and_22);  ge_14 = logical_and_22 = None
        convert_element_type_18: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_20, torch.int64);  add_20 = None
        convert_element_type_19: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_18, torch.int64);  add_18 = None
        full_default_29: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_23: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_23, convert_element_type_18, full_default_29);  convert_element_type_18 = full_default_29 = None
        view_28: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_23, [arg0_1, 3, 345, 456]);  where_23 = None
        full_default_30: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_23, convert_element_type_19, full_default_30);  convert_element_type_19 = full_default_30 = None
        view_29: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_24, [arg0_1, 3, 345, 456]);  where_24 = None
        full_default_31: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_32: i64[] = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_23, full_default_32, full_default_31);  logical_and_23 = full_default_32 = full_default_31 = None
        view_30: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_25, [arg0_1, 3, 345, 456]);  where_25 = None
        index_7: f32[s0, 3, 345, 456] = torch.ops.aten.index.Tensor(arg1_1, [view_5, view_6, view_29, view_28]);  view_29 = view_28 = None
        mul_30: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(index_7, view_30);  index_7 = view_30 = None
        add_21: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(sub_6, 1.0)
        mul_31: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(add_21, -0.75)
        sub_18: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_31, -3.75);  mul_31 = None
        mul_32: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_18, add_21);  sub_18 = None
        add_22: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_32, -6.0);  mul_32 = None
        mul_33: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(add_22, add_21);  add_22 = add_21 = None
        sub_19: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_33, -3.0);  mul_33 = None
        mul_34: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_6, 1.25)
        sub_20: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_34, 2.25);  mul_34 = None
        mul_35: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_20, sub_6);  sub_20 = None
        mul_36: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_35, sub_6);  mul_35 = None
        add_23: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_36, 1);  mul_36 = None
        sub_21: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(1.0, sub_6)
        mul_37: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_21, 1.25)
        sub_22: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_37, 2.25);  mul_37 = None
        mul_38: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_22, sub_21);  sub_22 = None
        mul_39: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_38, sub_21);  mul_38 = sub_21 = None
        add_24: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_39, 1);  mul_39 = None
        sub_23: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(2.0, sub_6)
        mul_40: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_23, -0.75)
        sub_24: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_40, -3.75);  mul_40 = None
        mul_41: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_24, sub_23);  sub_24 = None
        add_25: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_41, -6.0);  mul_41 = None
        mul_42: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(add_25, sub_23);  add_25 = sub_23 = None
        sub_25: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_42, -3.0);  mul_42 = None
        mul_43: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_27, sub_19);  mul_27 = sub_19 = None
        mul_44: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_28, add_23);  mul_28 = add_23 = None
        add_26: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
        mul_45: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_29, add_24);  mul_29 = add_24 = None
        add_27: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(add_26, mul_45);  add_26 = mul_45 = None
        mul_46: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_30, sub_25);  mul_30 = sub_25 = None
        add_28: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(add_27, mul_46);  add_27 = mul_46 = None
        add_29: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(floor_1, 1)
        sub_26: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(floor, 1)
        ge_16: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(sub_26, 0)
        lt_18: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(sub_26, 456)
        ge_17: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_29, 0)
        lt_19: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_29, 345)
        logical_and_24: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_17, lt_19);  ge_17 = lt_19 = None
        logical_and_25: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(lt_18, logical_and_24);  lt_18 = logical_and_24 = None
        logical_and_26: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_16, logical_and_25);  ge_16 = logical_and_25 = None
        convert_element_type_20: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(sub_26, torch.int64);  sub_26 = None
        convert_element_type_21: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_29, torch.int64)
        full_default_33: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_26, convert_element_type_20, full_default_33);  convert_element_type_20 = full_default_33 = None
        view_31: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_26, [arg0_1, 3, 345, 456]);  where_26 = None
        full_default_34: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_27: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_26, convert_element_type_21, full_default_34);  convert_element_type_21 = full_default_34 = None
        view_32: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_27, [arg0_1, 3, 345, 456]);  where_27 = None
        full_default_35: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_36: i64[] = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_28: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_26, full_default_36, full_default_35);  logical_and_26 = full_default_36 = full_default_35 = None
        view_33: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_28, [arg0_1, 3, 345, 456]);  where_28 = None
        index_8: f32[s0, 3, 345, 456] = torch.ops.aten.index.Tensor(arg1_1, [view_5, view_6, view_32, view_31]);  view_32 = view_31 = None
        mul_47: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(index_8, view_33);  index_8 = view_33 = None
        ge_18: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(floor, 0)
        lt_20: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(floor, 456)
        ge_19: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_29, 0)
        lt_21: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_29, 345)
        logical_and_27: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_19, lt_21);  ge_19 = lt_21 = None
        logical_and_28: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(lt_20, logical_and_27);  lt_20 = logical_and_27 = None
        logical_and_29: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_18, logical_and_28);  ge_18 = logical_and_28 = None
        convert_element_type_22: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(floor, torch.int64)
        convert_element_type_23: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_29, torch.int64)
        full_default_37: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_29: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_29, convert_element_type_22, full_default_37);  convert_element_type_22 = full_default_37 = None
        view_34: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_29, [arg0_1, 3, 345, 456]);  where_29 = None
        full_default_38: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_30: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_29, convert_element_type_23, full_default_38);  convert_element_type_23 = full_default_38 = None
        view_35: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_30, [arg0_1, 3, 345, 456]);  where_30 = None
        full_default_39: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_40: i64[] = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_31: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_29, full_default_40, full_default_39);  logical_and_29 = full_default_40 = full_default_39 = None
        view_36: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_31, [arg0_1, 3, 345, 456]);  where_31 = None
        index_9: f32[s0, 3, 345, 456] = torch.ops.aten.index.Tensor(arg1_1, [view_5, view_6, view_35, view_34]);  view_35 = view_34 = None
        mul_48: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(index_9, view_36);  index_9 = view_36 = None
        add_30: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(floor, 1)
        ge_20: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_30, 0)
        lt_22: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_30, 456)
        ge_21: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_29, 0)
        lt_23: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_29, 345)
        logical_and_30: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_21, lt_23);  ge_21 = lt_23 = None
        logical_and_31: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(lt_22, logical_and_30);  lt_22 = logical_and_30 = None
        logical_and_32: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_20, logical_and_31);  ge_20 = logical_and_31 = None
        convert_element_type_24: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_30, torch.int64);  add_30 = None
        convert_element_type_25: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_29, torch.int64)
        full_default_41: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_32: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_32, convert_element_type_24, full_default_41);  convert_element_type_24 = full_default_41 = None
        view_37: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_32, [arg0_1, 3, 345, 456]);  where_32 = None
        full_default_42: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_33: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_32, convert_element_type_25, full_default_42);  convert_element_type_25 = full_default_42 = None
        view_38: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_33, [arg0_1, 3, 345, 456]);  where_33 = None
        full_default_43: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_44: i64[] = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_34: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_32, full_default_44, full_default_43);  logical_and_32 = full_default_44 = full_default_43 = None
        view_39: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_34, [arg0_1, 3, 345, 456]);  where_34 = None
        index_10: f32[s0, 3, 345, 456] = torch.ops.aten.index.Tensor(arg1_1, [view_5, view_6, view_38, view_37]);  view_38 = view_37 = None
        mul_49: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(index_10, view_39);  index_10 = view_39 = None
        add_31: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(floor, 2)
        ge_22: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_31, 0)
        lt_24: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_31, 456)
        ge_23: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_29, 0)
        lt_25: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_29, 345)
        logical_and_33: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_23, lt_25);  ge_23 = lt_25 = None
        logical_and_34: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(lt_24, logical_and_33);  lt_24 = logical_and_33 = None
        logical_and_35: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_22, logical_and_34);  ge_22 = logical_and_34 = None
        convert_element_type_26: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_31, torch.int64);  add_31 = None
        convert_element_type_27: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_29, torch.int64);  add_29 = None
        full_default_45: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_35: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_35, convert_element_type_26, full_default_45);  convert_element_type_26 = full_default_45 = None
        view_40: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_35, [arg0_1, 3, 345, 456]);  where_35 = None
        full_default_46: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_36: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_35, convert_element_type_27, full_default_46);  convert_element_type_27 = full_default_46 = None
        view_41: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_36, [arg0_1, 3, 345, 456]);  where_36 = None
        full_default_47: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_48: i64[] = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_37: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_35, full_default_48, full_default_47);  logical_and_35 = full_default_48 = full_default_47 = None
        view_42: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_37, [arg0_1, 3, 345, 456]);  where_37 = None
        index_11: f32[s0, 3, 345, 456] = torch.ops.aten.index.Tensor(arg1_1, [view_5, view_6, view_41, view_40]);  view_41 = view_40 = None
        mul_50: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(index_11, view_42);  index_11 = view_42 = None
        add_32: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(sub_6, 1.0)
        mul_51: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(add_32, -0.75)
        sub_27: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_51, -3.75);  mul_51 = None
        mul_52: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_27, add_32);  sub_27 = None
        add_33: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_52, -6.0);  mul_52 = None
        mul_53: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(add_33, add_32);  add_33 = add_32 = None
        sub_28: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_53, -3.0);  mul_53 = None
        mul_54: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_6, 1.25)
        sub_29: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_54, 2.25);  mul_54 = None
        mul_55: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_29, sub_6);  sub_29 = None
        mul_56: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_55, sub_6);  mul_55 = None
        add_34: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_56, 1);  mul_56 = None
        sub_30: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(1.0, sub_6)
        mul_57: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_30, 1.25)
        sub_31: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_57, 2.25);  mul_57 = None
        mul_58: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_31, sub_30);  sub_31 = None
        mul_59: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_58, sub_30);  mul_58 = sub_30 = None
        add_35: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_59, 1);  mul_59 = None
        sub_32: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(2.0, sub_6)
        mul_60: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_32, -0.75)
        sub_33: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_60, -3.75);  mul_60 = None
        mul_61: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_33, sub_32);  sub_33 = None
        add_36: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_61, -6.0);  mul_61 = None
        mul_62: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(add_36, sub_32);  add_36 = sub_32 = None
        sub_34: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_62, -3.0);  mul_62 = None
        mul_63: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_47, sub_28);  mul_47 = sub_28 = None
        mul_64: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_48, add_34);  mul_48 = add_34 = None
        add_37: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_63, mul_64);  mul_63 = mul_64 = None
        mul_65: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_49, add_35);  mul_49 = add_35 = None
        add_38: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(add_37, mul_65);  add_37 = mul_65 = None
        mul_66: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_50, sub_34);  mul_50 = sub_34 = None
        add_39: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(add_38, mul_66);  add_38 = mul_66 = None
        add_40: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(floor_1, 2);  floor_1 = None
        sub_35: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(floor, 1)
        ge_24: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(sub_35, 0)
        lt_26: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(sub_35, 456)
        ge_25: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_40, 0)
        lt_27: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_40, 345)
        logical_and_36: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_25, lt_27);  ge_25 = lt_27 = None
        logical_and_37: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(lt_26, logical_and_36);  lt_26 = logical_and_36 = None
        logical_and_38: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_24, logical_and_37);  ge_24 = logical_and_37 = None
        convert_element_type_28: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(sub_35, torch.int64);  sub_35 = None
        convert_element_type_29: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_40, torch.int64)
        full_default_49: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_38: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_38, convert_element_type_28, full_default_49);  convert_element_type_28 = full_default_49 = None
        view_43: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_38, [arg0_1, 3, 345, 456]);  where_38 = None
        full_default_50: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_39: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_38, convert_element_type_29, full_default_50);  convert_element_type_29 = full_default_50 = None
        view_44: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_39, [arg0_1, 3, 345, 456]);  where_39 = None
        full_default_51: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_52: i64[] = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_40: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_38, full_default_52, full_default_51);  logical_and_38 = full_default_52 = full_default_51 = None
        view_45: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_40, [arg0_1, 3, 345, 456]);  where_40 = None
        index_12: f32[s0, 3, 345, 456] = torch.ops.aten.index.Tensor(arg1_1, [view_5, view_6, view_44, view_43]);  view_44 = view_43 = None
        mul_67: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(index_12, view_45);  index_12 = view_45 = None
        ge_26: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(floor, 0)
        lt_28: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(floor, 456)
        ge_27: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_40, 0)
        lt_29: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_40, 345)
        logical_and_39: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_27, lt_29);  ge_27 = lt_29 = None
        logical_and_40: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(lt_28, logical_and_39);  lt_28 = logical_and_39 = None
        logical_and_41: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_26, logical_and_40);  ge_26 = logical_and_40 = None
        convert_element_type_30: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(floor, torch.int64)
        convert_element_type_31: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_40, torch.int64)
        full_default_53: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_41: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_41, convert_element_type_30, full_default_53);  convert_element_type_30 = full_default_53 = None
        view_46: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_41, [arg0_1, 3, 345, 456]);  where_41 = None
        full_default_54: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_42: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_41, convert_element_type_31, full_default_54);  convert_element_type_31 = full_default_54 = None
        view_47: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_42, [arg0_1, 3, 345, 456]);  where_42 = None
        full_default_55: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_56: i64[] = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_43: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_41, full_default_56, full_default_55);  logical_and_41 = full_default_56 = full_default_55 = None
        view_48: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_43, [arg0_1, 3, 345, 456]);  where_43 = None
        index_13: f32[s0, 3, 345, 456] = torch.ops.aten.index.Tensor(arg1_1, [view_5, view_6, view_47, view_46]);  view_47 = view_46 = None
        mul_68: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(index_13, view_48);  index_13 = view_48 = None
        add_41: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(floor, 1)
        ge_28: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_41, 0)
        lt_30: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_41, 456)
        ge_29: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_40, 0)
        lt_31: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_40, 345)
        logical_and_42: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_29, lt_31);  ge_29 = lt_31 = None
        logical_and_43: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(lt_30, logical_and_42);  lt_30 = logical_and_42 = None
        logical_and_44: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_28, logical_and_43);  ge_28 = logical_and_43 = None
        convert_element_type_32: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_41, torch.int64);  add_41 = None
        convert_element_type_33: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_40, torch.int64)
        full_default_57: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_44: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_44, convert_element_type_32, full_default_57);  convert_element_type_32 = full_default_57 = None
        view_49: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_44, [arg0_1, 3, 345, 456]);  where_44 = None
        full_default_58: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_45: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_44, convert_element_type_33, full_default_58);  convert_element_type_33 = full_default_58 = None
        view_50: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_45, [arg0_1, 3, 345, 456]);  where_45 = None
        full_default_59: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_60: i64[] = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_46: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_44, full_default_60, full_default_59);  logical_and_44 = full_default_60 = full_default_59 = None
        view_51: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_46, [arg0_1, 3, 345, 456]);  where_46 = None
        index_14: f32[s0, 3, 345, 456] = torch.ops.aten.index.Tensor(arg1_1, [view_5, view_6, view_50, view_49]);  view_50 = view_49 = None
        mul_69: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(index_14, view_51);  index_14 = view_51 = None
        add_42: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(floor, 2);  floor = None
        ge_30: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_42, 0)
        lt_32: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_42, 456)
        ge_31: b8[s0, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_40, 0)
        lt_33: b8[s0, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_40, 345)
        logical_and_45: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_31, lt_33);  ge_31 = lt_33 = None
        logical_and_46: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(lt_32, logical_and_45);  lt_32 = logical_and_45 = None
        logical_and_47: b8[s0, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_30, logical_and_46);  ge_30 = logical_and_46 = None
        convert_element_type_34: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_42, torch.int64);  add_42 = None
        convert_element_type_35: i64[s0, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_40, torch.int64);  add_40 = None
        full_default_61: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_47: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_47, convert_element_type_34, full_default_61);  convert_element_type_34 = full_default_61 = None
        view_52: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_47, [arg0_1, 3, 345, 456]);  where_47 = None
        full_default_62: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_48: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_47, convert_element_type_35, full_default_62);  convert_element_type_35 = full_default_62 = None
        view_53: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_48, [arg0_1, 3, 345, 456]);  where_48 = None
        full_default_63: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_64: i64[] = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_49: i64[s0, 3, 345, 456] = torch.ops.aten.where.self(logical_and_47, full_default_64, full_default_63);  logical_and_47 = full_default_64 = full_default_63 = None
        view_54: i64[s0, 3, 345, 456] = torch.ops.aten.view.default(where_49, [arg0_1, 3, 345, 456]);  where_49 = arg0_1 = None
        index_15: f32[s0, 3, 345, 456] = torch.ops.aten.index.Tensor(arg1_1, [view_5, view_6, view_53, view_52]);  arg1_1 = view_5 = view_6 = view_53 = view_52 = None
        mul_70: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(index_15, view_54);  index_15 = view_54 = None
        add_43: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(sub_6, 1.0)
        mul_71: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(add_43, -0.75)
        sub_36: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_71, -3.75);  mul_71 = None
        mul_72: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_36, add_43);  sub_36 = None
        add_44: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_72, -6.0);  mul_72 = None
        mul_73: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(add_44, add_43);  add_44 = add_43 = None
        sub_37: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_73, -3.0);  mul_73 = None
        mul_74: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_6, 1.25)
        sub_38: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_74, 2.25);  mul_74 = None
        mul_75: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_38, sub_6);  sub_38 = None
        mul_76: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_75, sub_6);  mul_75 = None
        add_45: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_76, 1);  mul_76 = None
        sub_39: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(1.0, sub_6)
        mul_77: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_39, 1.25)
        sub_40: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_77, 2.25);  mul_77 = None
        mul_78: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_40, sub_39);  sub_40 = None
        mul_79: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_78, sub_39);  mul_78 = sub_39 = None
        add_46: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_79, 1);  mul_79 = None
        sub_41: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(2.0, sub_6);  sub_6 = None
        mul_80: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_41, -0.75)
        sub_42: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_80, -3.75);  mul_80 = None
        mul_81: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_42, sub_41);  sub_42 = None
        add_47: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_81, -6.0);  mul_81 = None
        mul_82: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(add_47, sub_41);  add_47 = sub_41 = None
        sub_43: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_82, -3.0);  mul_82 = None
        mul_83: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_67, sub_37);  mul_67 = sub_37 = None
        mul_84: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_68, add_45);  mul_68 = add_45 = None
        add_48: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_83, mul_84);  mul_83 = mul_84 = None
        mul_85: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_69, add_46);  mul_69 = add_46 = None
        add_49: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(add_48, mul_85);  add_48 = mul_85 = None
        mul_86: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_70, sub_43);  mul_70 = sub_43 = None
        add_50: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(add_49, mul_86);  add_49 = mul_86 = None
        add_51: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(sub_7, 1.0)
        mul_87: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(add_51, -0.75)
        sub_44: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_87, -3.75);  mul_87 = None
        mul_88: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_44, add_51);  sub_44 = None
        add_52: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_88, -6.0);  mul_88 = None
        mul_89: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(add_52, add_51);  add_52 = add_51 = None
        sub_45: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_89, -3.0);  mul_89 = None
        mul_90: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_7, 1.25)
        sub_46: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_90, 2.25);  mul_90 = None
        mul_91: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_46, sub_7);  sub_46 = None
        mul_92: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_91, sub_7);  mul_91 = None
        add_53: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_92, 1);  mul_92 = None
        sub_47: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(1.0, sub_7)
        mul_93: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_47, 1.25)
        sub_48: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_93, 2.25);  mul_93 = None
        mul_94: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_48, sub_47);  sub_48 = None
        mul_95: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(mul_94, sub_47);  mul_94 = sub_47 = None
        add_54: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_95, 1);  mul_95 = None
        sub_49: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(2.0, sub_7);  sub_7 = None
        mul_96: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_49, -0.75)
        sub_50: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_96, -3.75);  mul_96 = None
        mul_97: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_50, sub_49);  sub_50 = None
        add_55: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_97, -6.0);  mul_97 = None
        mul_98: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(add_55, sub_49);  add_55 = sub_49 = None
        sub_51: f32[s0, 3, 345, 456] = torch.ops.aten.sub.Tensor(mul_98, -3.0);  mul_98 = None
        mul_99: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(add_17, sub_45);  add_17 = sub_45 = None
        mul_100: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(add_28, add_53);  add_28 = add_53 = None
        add_56: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
        mul_101: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(add_39, add_54);  add_39 = add_54 = None
        add_57: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(add_56, mul_101);  add_56 = mul_101 = None
        mul_102: f32[s0, 3, 345, 456] = torch.ops.aten.mul.Tensor(add_50, sub_51);  add_50 = sub_51 = None
        add_58: f32[s0, 3, 345, 456] = torch.ops.aten.add.Tensor(add_57, mul_102);  add_57 = mul_102 = None
        return (add_58,)
        