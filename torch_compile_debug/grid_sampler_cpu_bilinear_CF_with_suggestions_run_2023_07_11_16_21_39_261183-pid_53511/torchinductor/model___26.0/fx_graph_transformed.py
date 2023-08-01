class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: f32[2, 3, 345, 456], arg1_1: f32[2, 2, 3]):
        # File: check_affine_grid_sampler.py:26, code: output = grid_sample(img, grid, align_corners=align_corners, mode=mode)
        iota_2: i64[2] = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_4: i64[2, 1, 1, 1] = torch.ops.aten.reshape.default(iota_2, [2, 1, 1, 1]);  iota_2 = None
        iota_3: i64[3] = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_5: i64[1, 3, 1, 1] = torch.ops.aten.reshape.default(iota_3, [1, 3, 1, 1]);  iota_3 = None
        
        # File: check_affine_grid_sampler.py:24, code: grid = affine_grid(theta, size=(n, c, h, w), align_corners=align_corners)
        iota: i64[456] = torch.ops.prims.iota.default(456, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        lt: b8[456] = torch.ops.aten.lt.Scalar(iota, 228.0)
        convert_element_type: f32[456] = torch.ops.prims.convert_element_type.default(iota, torch.float32)
        mul: f32[456] = torch.ops.aten.mul.Tensor(convert_element_type, 0.004385964912280702);  convert_element_type = None
        add: f32[456] = torch.ops.aten.add.Tensor(mul, -0.9978070175438597);  mul = None
        sub: i64[456] = torch.ops.aten.sub.Tensor(455, iota);  iota = None
        convert_element_type_1: f32[456] = torch.ops.prims.convert_element_type.default(sub, torch.float32);  sub = None
        mul_1: f32[456] = torch.ops.aten.mul.Tensor(convert_element_type_1, 0.004385964912280702);  convert_element_type_1 = None
        sub_1: f32[456] = torch.ops.aten.sub.Tensor(0.9978070175438597, mul_1);  mul_1 = None
        where: f32[456] = torch.ops.aten.where.self(lt, add, sub_1);  lt = add = sub_1 = None
        view: f32[1, 456, 1] = torch.ops.aten.reshape.default(where, [1, 456, 1]);  where = None
        constant_pad_nd: f32[1, 456, 3] = torch.ops.aten.constant_pad_nd.default(view, [0, 2], 0.0);  view = None
        iota_1: i64[345] = torch.ops.prims.iota.default(345, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        lt_1: b8[345] = torch.ops.aten.lt.Scalar(iota_1, 172.5)
        convert_element_type_2: f32[345] = torch.ops.prims.convert_element_type.default(iota_1, torch.float32)
        mul_2: f32[345] = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.005797101449275362);  convert_element_type_2 = None
        add_1: f32[345] = torch.ops.aten.add.Tensor(mul_2, -0.9971014492753624);  mul_2 = None
        sub_2: i64[345] = torch.ops.aten.sub.Tensor(344, iota_1);  iota_1 = None
        convert_element_type_3: f32[345] = torch.ops.prims.convert_element_type.default(sub_2, torch.float32);  sub_2 = None
        mul_3: f32[345] = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.005797101449275362);  convert_element_type_3 = None
        sub_3: f32[345] = torch.ops.aten.sub.Tensor(0.9971014492753624, mul_3);  mul_3 = None
        where_1: f32[345] = torch.ops.aten.where.self(lt_1, add_1, sub_3);  lt_1 = add_1 = sub_3 = None
        view_1: f32[345, 1, 1] = torch.ops.aten.reshape.default(where_1, [345, 1, 1]);  where_1 = None
        constant_pad_nd_1: f32[345, 1, 3] = torch.ops.aten.constant_pad_nd.default(view_1, [1, 1], 0.0);  view_1 = None
        add_2: f32[345, 456, 3] = torch.ops.aten.add.Tensor(constant_pad_nd, constant_pad_nd_1);  constant_pad_nd = constant_pad_nd_1 = None
        full_default: f32[1, 1, 1] = torch.ops.aten.full.default([1, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        constant_pad_nd_2: f32[1, 1, 3] = torch.ops.aten.constant_pad_nd.default(full_default, [2, 0], 0.0);  full_default = None
        add_3: f32[345, 456, 3] = torch.ops.aten.add.Tensor(add_2, constant_pad_nd_2);  add_2 = constant_pad_nd_2 = None
        unsqueeze: f32[345, 456, 3, 1] = torch.ops.aten.unsqueeze.default(add_3, -1);  add_3 = None
        permute: f32[2, 3, 2] = torch.ops.aten.permute.default(arg1_1, [0, 2, 1]);  arg1_1 = None
        view_2: f32[2, 1, 1, 3, 2] = torch.ops.aten.reshape.default(permute, [2, 1, 1, 3, 2]);  permute = None
        mul_4: f32[2, 345, 456, 3, 2] = torch.ops.aten.mul.Tensor(unsqueeze, view_2);  unsqueeze = view_2 = None
        sum_1: f32[2, 345, 456, 2] = torch.ops.aten.sum.dim_IntList(mul_4, [-2]);  mul_4 = None
        
        # File: check_affine_grid_sampler.py:26, code: output = grid_sample(img, grid, align_corners=align_corners, mode=mode)
        view_3: f32[2, 1, 345, 456, 2] = torch.ops.aten.reshape.default(sum_1, [2, 1, 345, 456, 2]);  sum_1 = None
        expand: f32[2, 3, 345, 456, 2] = torch.ops.aten.expand.default(view_3, [2, 3, 345, 456, 2]);  view_3 = None
        select: f32[2, 3, 345, 456] = torch.ops.aten.select.int(expand, 4, 0)
        mul_5: f32[2, 3, 345, 456] = torch.ops.aten.mul.Tensor(select, 228.0);  select = None
        add_4: f32[2, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_5, 227.5);  mul_5 = None
        floor: f32[2, 3, 345, 456] = torch.ops.aten.floor.default(add_4)
        ge: b8[2, 3, 345, 456] = torch.ops.aten.ge.Scalar(floor, 0)
        lt_2: b8[2, 3, 345, 456] = torch.ops.aten.lt.Scalar(floor, 456)
        select_1: f32[2, 3, 345, 456] = torch.ops.aten.select.int(expand, 4, 1);  expand = None
        mul_6: f32[2, 3, 345, 456] = torch.ops.aten.mul.Tensor(select_1, 172.5);  select_1 = None
        add_5: f32[2, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_6, 172.0);  mul_6 = None
        floor_1: f32[2, 3, 345, 456] = torch.ops.aten.floor.default(add_5)
        ge_1: b8[2, 3, 345, 456] = torch.ops.aten.ge.Scalar(floor_1, 0)
        lt_3: b8[2, 3, 345, 456] = torch.ops.aten.lt.Scalar(floor_1, 345)
        logical_and: b8[2, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_1, lt_3);  ge_1 = lt_3 = None
        logical_and_1: b8[2, 3, 345, 456] = torch.ops.aten.logical_and.default(lt_2, logical_and);  lt_2 = logical_and = None
        logical_and_2: b8[2, 3, 345, 456] = torch.ops.aten.logical_and.default(ge, logical_and_1);  ge = logical_and_1 = None
        convert_element_type_5: i64[2, 3, 345, 456] = torch.ops.prims.convert_element_type.default(floor_1, torch.int64)
        full_default_2: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_3: i64[2, 3, 345, 456] = torch.ops.aten.where.self(logical_and_2, convert_element_type_5, full_default_2);  convert_element_type_5 = full_default_2 = None
        convert_element_type_4: i64[2, 3, 345, 456] = torch.ops.prims.convert_element_type.default(floor, torch.int64)
        full_default_1: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_2: i64[2, 3, 345, 456] = torch.ops.aten.where.self(logical_and_2, convert_element_type_4, full_default_1);  convert_element_type_4 = full_default_1 = None
        index: f32[2, 3, 345, 456] = torch.ops.aten.index.Tensor(arg0_1, [view_4, view_5, where_3, where_2]);  where_3 = where_2 = None
        add_6: f32[2, 3, 345, 456] = torch.ops.aten.add.Tensor(floor, 1)
        sub_4: f32[2, 3, 345, 456] = torch.ops.aten.sub.Tensor(add_6, add_4)
        add_7: f32[2, 3, 345, 456] = torch.ops.aten.add.Tensor(floor_1, 1)
        sub_5: f32[2, 3, 345, 456] = torch.ops.aten.sub.Tensor(add_7, add_5)
        mul_7: f32[2, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_4, sub_5);  sub_4 = sub_5 = None
        full_default_3: f32[] = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_4: f32[2, 3, 345, 456] = torch.ops.aten.where.self(logical_and_2, mul_7, full_default_3);  logical_and_2 = mul_7 = full_default_3 = None
        mul_11: f32[2, 3, 345, 456] = torch.ops.aten.mul.Tensor(index, where_4);  index = where_4 = None
        ge_2: b8[2, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_6, 0)
        lt_4: b8[2, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_6, 456)
        ge_3: b8[2, 3, 345, 456] = torch.ops.aten.ge.Scalar(floor_1, 0)
        lt_5: b8[2, 3, 345, 456] = torch.ops.aten.lt.Scalar(floor_1, 345)
        logical_and_3: b8[2, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_3, lt_5);  ge_3 = lt_5 = None
        logical_and_4: b8[2, 3, 345, 456] = torch.ops.aten.logical_and.default(lt_4, logical_and_3);  lt_4 = logical_and_3 = None
        logical_and_5: b8[2, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_2, logical_and_4);  ge_2 = logical_and_4 = None
        convert_element_type_7: i64[2, 3, 345, 456] = torch.ops.prims.convert_element_type.default(floor_1, torch.int64)
        full_default_5: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_6: i64[2, 3, 345, 456] = torch.ops.aten.where.self(logical_and_5, convert_element_type_7, full_default_5);  convert_element_type_7 = full_default_5 = None
        convert_element_type_6: i64[2, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_6, torch.int64)
        full_default_4: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_5: i64[2, 3, 345, 456] = torch.ops.aten.where.self(logical_and_5, convert_element_type_6, full_default_4);  convert_element_type_6 = full_default_4 = None
        index_1: f32[2, 3, 345, 456] = torch.ops.aten.index.Tensor(arg0_1, [view_4, view_5, where_6, where_5]);  where_6 = where_5 = None
        sub_6: f32[2, 3, 345, 456] = torch.ops.aten.sub.Tensor(add_4, floor)
        sub_7: f32[2, 3, 345, 456] = torch.ops.aten.sub.Tensor(add_7, add_5)
        mul_8: f32[2, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_6, sub_7);  sub_6 = sub_7 = None
        full_default_6: f32[] = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_7: f32[2, 3, 345, 456] = torch.ops.aten.where.self(logical_and_5, mul_8, full_default_6);  logical_and_5 = mul_8 = full_default_6 = None
        mul_12: f32[2, 3, 345, 456] = torch.ops.aten.mul.Tensor(index_1, where_7);  index_1 = where_7 = None
        add_8: f32[2, 3, 345, 456] = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        ge_4: b8[2, 3, 345, 456] = torch.ops.aten.ge.Scalar(floor, 0)
        lt_6: b8[2, 3, 345, 456] = torch.ops.aten.lt.Scalar(floor, 456)
        ge_5: b8[2, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_7, 0)
        lt_7: b8[2, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_7, 345)
        logical_and_6: b8[2, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_5, lt_7);  ge_5 = lt_7 = None
        logical_and_7: b8[2, 3, 345, 456] = torch.ops.aten.logical_and.default(lt_6, logical_and_6);  lt_6 = logical_and_6 = None
        logical_and_8: b8[2, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_4, logical_and_7);  ge_4 = logical_and_7 = None
        convert_element_type_9: i64[2, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_7, torch.int64)
        full_default_8: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_9: i64[2, 3, 345, 456] = torch.ops.aten.where.self(logical_and_8, convert_element_type_9, full_default_8);  convert_element_type_9 = full_default_8 = None
        convert_element_type_8: i64[2, 3, 345, 456] = torch.ops.prims.convert_element_type.default(floor, torch.int64)
        full_default_7: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_8: i64[2, 3, 345, 456] = torch.ops.aten.where.self(logical_and_8, convert_element_type_8, full_default_7);  convert_element_type_8 = full_default_7 = None
        index_2: f32[2, 3, 345, 456] = torch.ops.aten.index.Tensor(arg0_1, [view_4, view_5, where_9, where_8]);  where_9 = where_8 = None
        sub_8: f32[2, 3, 345, 456] = torch.ops.aten.sub.Tensor(add_6, add_4)
        sub_9: f32[2, 3, 345, 456] = torch.ops.aten.sub.Tensor(add_5, floor_1)
        mul_9: f32[2, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_8, sub_9);  sub_8 = sub_9 = None
        full_default_9: f32[] = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_10: f32[2, 3, 345, 456] = torch.ops.aten.where.self(logical_and_8, mul_9, full_default_9);  logical_and_8 = mul_9 = full_default_9 = None
        mul_13: f32[2, 3, 345, 456] = torch.ops.aten.mul.Tensor(index_2, where_10);  index_2 = where_10 = None
        add_9: f32[2, 3, 345, 456] = torch.ops.aten.add.Tensor(add_8, mul_13);  add_8 = mul_13 = None
        ge_6: b8[2, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_6, 0)
        lt_8: b8[2, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_6, 456)
        ge_7: b8[2, 3, 345, 456] = torch.ops.aten.ge.Scalar(add_7, 0)
        lt_9: b8[2, 3, 345, 456] = torch.ops.aten.lt.Scalar(add_7, 345)
        logical_and_9: b8[2, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_7, lt_9);  ge_7 = lt_9 = None
        logical_and_10: b8[2, 3, 345, 456] = torch.ops.aten.logical_and.default(lt_8, logical_and_9);  lt_8 = logical_and_9 = None
        logical_and_11: b8[2, 3, 345, 456] = torch.ops.aten.logical_and.default(ge_6, logical_and_10);  ge_6 = logical_and_10 = None
        convert_element_type_11: i64[2, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_7, torch.int64);  add_7 = None
        full_default_11: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_12: i64[2, 3, 345, 456] = torch.ops.aten.where.self(logical_and_11, convert_element_type_11, full_default_11);  convert_element_type_11 = full_default_11 = None
        convert_element_type_10: i64[2, 3, 345, 456] = torch.ops.prims.convert_element_type.default(add_6, torch.int64);  add_6 = None
        full_default_10: i64[] = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_11: i64[2, 3, 345, 456] = torch.ops.aten.where.self(logical_and_11, convert_element_type_10, full_default_10);  convert_element_type_10 = full_default_10 = None
        index_3: f32[2, 3, 345, 456] = torch.ops.aten.index.Tensor(arg0_1, [view_4, view_5, where_12, where_11]);  arg0_1 = view_4 = view_5 = where_12 = where_11 = None
        sub_10: f32[2, 3, 345, 456] = torch.ops.aten.sub.Tensor(add_4, floor);  add_4 = floor = None
        sub_11: f32[2, 3, 345, 456] = torch.ops.aten.sub.Tensor(add_5, floor_1);  add_5 = floor_1 = None
        mul_10: f32[2, 3, 345, 456] = torch.ops.aten.mul.Tensor(sub_10, sub_11);  sub_10 = sub_11 = None
        full_default_12: f32[] = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_13: f32[2, 3, 345, 456] = torch.ops.aten.where.self(logical_and_11, mul_10, full_default_12);  logical_and_11 = mul_10 = full_default_12 = None
        mul_14: f32[2, 3, 345, 456] = torch.ops.aten.mul.Tensor(index_3, where_13);  index_3 = where_13 = None
        add_10: f32[2, 3, 345, 456] = torch.ops.aten.add.Tensor(add_9, mul_14);  add_9 = mul_14 = None
        return (add_10,)
        