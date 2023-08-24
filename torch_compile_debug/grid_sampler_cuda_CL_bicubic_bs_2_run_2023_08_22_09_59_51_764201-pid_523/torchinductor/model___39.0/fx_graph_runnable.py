
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config


torch._functorch.config.debug_partitioner = True


isolate_fails_code_str = None



# torch version: 2.1.0a0+gitafe1ff4
# torch cuda version: 11.7
# torch git version: afe1ff451e9c5a645b570a922ccf879653d00b19


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2022 NVIDIA Corporation 
# Built on Wed_Jun__8_16:49:14_PDT_2022 
# Cuda compilation tools, release 11.7, V11.7.99 
# Build cuda_11.7.r11.7/compiler.31442593_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 4090 : 1 
# NVIDIA GeForce GTX 1080 Ti : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1):
        iota = torch.ops.prims.iota.default(456, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        lt = torch.ops.aten.lt.Scalar(iota, 228.0)
        convert_element_type = torch.ops.prims.convert_element_type.default(iota, torch.float32)
        mul = torch.ops.aten.mul.Tensor(convert_element_type, 0.004385964912280702);  convert_element_type = None
        add = torch.ops.aten.add.Tensor(mul, -0.9978070175438597);  mul = None
        sub = torch.ops.aten.sub.Tensor(455, iota);  iota = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(sub, torch.float32);  sub = None
        mul_1 = torch.ops.aten.mul.Tensor(convert_element_type_1, 0.004385964912280702);  convert_element_type_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(0.9978070175438597, mul_1);  mul_1 = None
        where = torch.ops.aten.where.self(lt, add, sub_1);  lt = add = sub_1 = None
        view = torch.ops.aten.view.default(where, [1, 456, 1]);  where = None
        iota_1 = torch.ops.prims.iota.default(345, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        lt_1 = torch.ops.aten.lt.Scalar(iota_1, 172.5)
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(iota_1, torch.float32)
        mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_2, 0.005797101449275362);  convert_element_type_2 = None
        add_1 = torch.ops.aten.add.Tensor(mul_2, -0.9971014492753624);  mul_2 = None
        sub_2 = torch.ops.aten.sub.Tensor(344, iota_1);  iota_1 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(sub_2, torch.float32);  sub_2 = None
        mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_3, 0.005797101449275362);  convert_element_type_3 = None
        sub_3 = torch.ops.aten.sub.Tensor(0.9971014492753624, mul_3);  mul_3 = None
        where_1 = torch.ops.aten.where.self(lt_1, add_1, sub_3);  lt_1 = add_1 = sub_3 = None
        view_1 = torch.ops.aten.view.default(where_1, [345, 1, 1]);  where_1 = None
        full_default = torch.ops.aten.full.default([1, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        constant_pad_nd = torch.ops.aten.constant_pad_nd.default(view, [0, 2], 0.0);  view = None
        constant_pad_nd_1 = torch.ops.aten.constant_pad_nd.default(view_1, [1, 1], 0.0);  view_1 = None
        constant_pad_nd_2 = torch.ops.aten.constant_pad_nd.default(full_default, [2, 0], 0.0);  full_default = None
        add_2 = torch.ops.aten.add.Tensor(constant_pad_nd, constant_pad_nd_1);  constant_pad_nd = constant_pad_nd_1 = None
        add_3 = torch.ops.aten.add.Tensor(add_2, constant_pad_nd_2);  add_2 = constant_pad_nd_2 = None
        view_2 = torch.ops.aten.view.default(add_3, [-1, 3, 1]);  add_3 = None
        permute = torch.ops.aten.permute.default(arg1_1, [0, 2, 1]);  arg1_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(permute, 1);  permute = None
        mul_4 = torch.ops.aten.mul.Tensor(view_2, unsqueeze);  view_2 = unsqueeze = None
        sum_1 = torch.ops.aten.sum.dim_IntList(mul_4, [-2]);  mul_4 = None
        view_3 = torch.ops.aten.view.default(sum_1, [2, 345, 456, 2]);  sum_1 = None
        view_4 = torch.ops.aten.view.default(view_3, [2, 1, 345, 456, 2]);  view_3 = None
        expand = torch.ops.aten.expand.default(view_4, [2, 3, 345, 456, 2]);  view_4 = None
        iota_2 = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_5 = torch.ops.aten.view.default(iota_2, [2, 1, 1, 1]);  iota_2 = None
        iota_3 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_6 = torch.ops.aten.view.default(iota_3, [1, 3, 1, 1]);  iota_3 = None
        select = torch.ops.aten.select.int(expand, 4, 0)
        select_1 = torch.ops.aten.select.int(expand, 4, 1);  expand = None
        mul_5 = torch.ops.aten.mul.Tensor(select, 228.0);  select = None
        add_4 = torch.ops.aten.add.Tensor(mul_5, 227.5);  mul_5 = None
        mul_6 = torch.ops.aten.mul.Tensor(select_1, 172.5);  select_1 = None
        add_5 = torch.ops.aten.add.Tensor(mul_6, 172.0);  mul_6 = None
        floor = torch.ops.aten.floor.default(add_4)
        floor_1 = torch.ops.aten.floor.default(add_5)
        sub_4 = torch.ops.aten.sub.Tensor(add_4, floor);  add_4 = None
        sub_5 = torch.ops.aten.sub.Tensor(add_5, floor_1);  add_5 = None
        add_6 = torch.ops.aten.add.Tensor(floor_1, -1)
        sub_6 = torch.ops.aten.sub.Tensor(floor, 1)
        ge = torch.ops.aten.ge.Scalar(sub_6, 0)
        lt_2 = torch.ops.aten.lt.Scalar(sub_6, 456)
        ge_1 = torch.ops.aten.ge.Scalar(add_6, 0)
        lt_3 = torch.ops.aten.lt.Scalar(add_6, 345)
        logical_and = torch.ops.aten.logical_and.default(ge_1, lt_3);  ge_1 = lt_3 = None
        logical_and_1 = torch.ops.aten.logical_and.default(lt_2, logical_and);  lt_2 = logical_and = None
        logical_and_2 = torch.ops.aten.logical_and.default(ge, logical_and_1);  ge = logical_and_1 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(sub_6, torch.int64);  sub_6 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(add_6, torch.int64)
        full_default_1 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2 = torch.ops.aten.where.self(logical_and_2, convert_element_type_4, full_default_1);  convert_element_type_4 = full_default_1 = None
        full_default_2 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3 = torch.ops.aten.where.self(logical_and_2, convert_element_type_5, full_default_2);  convert_element_type_5 = full_default_2 = None
        full_default_3 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4 = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4 = torch.ops.aten.where.self(logical_and_2, full_default_4, full_default_3);  logical_and_2 = full_default_4 = full_default_3 = None
        index = torch.ops.aten.index.Tensor(arg0_1, [view_5, view_6, where_3, where_2]);  where_3 = where_2 = None
        mul_7 = torch.ops.aten.mul.Tensor(index, where_4);  index = where_4 = None
        ge_2 = torch.ops.aten.ge.Scalar(floor, 0)
        lt_4 = torch.ops.aten.lt.Scalar(floor, 456)
        ge_3 = torch.ops.aten.ge.Scalar(add_6, 0)
        lt_5 = torch.ops.aten.lt.Scalar(add_6, 345)
        logical_and_3 = torch.ops.aten.logical_and.default(ge_3, lt_5);  ge_3 = lt_5 = None
        logical_and_4 = torch.ops.aten.logical_and.default(lt_4, logical_and_3);  lt_4 = logical_and_3 = None
        logical_and_5 = torch.ops.aten.logical_and.default(ge_2, logical_and_4);  ge_2 = logical_and_4 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(floor, torch.int64)
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(add_6, torch.int64)
        full_default_5 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5 = torch.ops.aten.where.self(logical_and_5, convert_element_type_6, full_default_5);  convert_element_type_6 = full_default_5 = None
        full_default_6 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_6 = torch.ops.aten.where.self(logical_and_5, convert_element_type_7, full_default_6);  convert_element_type_7 = full_default_6 = None
        full_default_7 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8 = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7 = torch.ops.aten.where.self(logical_and_5, full_default_8, full_default_7);  logical_and_5 = full_default_8 = full_default_7 = None
        index_1 = torch.ops.aten.index.Tensor(arg0_1, [view_5, view_6, where_6, where_5]);  where_6 = where_5 = None
        mul_8 = torch.ops.aten.mul.Tensor(index_1, where_7);  index_1 = where_7 = None
        add_7 = torch.ops.aten.add.Tensor(floor, 1)
        ge_4 = torch.ops.aten.ge.Scalar(add_7, 0)
        lt_6 = torch.ops.aten.lt.Scalar(add_7, 456)
        ge_5 = torch.ops.aten.ge.Scalar(add_6, 0)
        lt_7 = torch.ops.aten.lt.Scalar(add_6, 345)
        logical_and_6 = torch.ops.aten.logical_and.default(ge_5, lt_7);  ge_5 = lt_7 = None
        logical_and_7 = torch.ops.aten.logical_and.default(lt_6, logical_and_6);  lt_6 = logical_and_6 = None
        logical_and_8 = torch.ops.aten.logical_and.default(ge_4, logical_and_7);  ge_4 = logical_and_7 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(add_7, torch.int64);  add_7 = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(add_6, torch.int64)
        full_default_9 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8 = torch.ops.aten.where.self(logical_and_8, convert_element_type_8, full_default_9);  convert_element_type_8 = full_default_9 = None
        full_default_10 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_9 = torch.ops.aten.where.self(logical_and_8, convert_element_type_9, full_default_10);  convert_element_type_9 = full_default_10 = None
        full_default_11 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12 = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10 = torch.ops.aten.where.self(logical_and_8, full_default_12, full_default_11);  logical_and_8 = full_default_12 = full_default_11 = None
        index_2 = torch.ops.aten.index.Tensor(arg0_1, [view_5, view_6, where_9, where_8]);  where_9 = where_8 = None
        mul_9 = torch.ops.aten.mul.Tensor(index_2, where_10);  index_2 = where_10 = None
        add_8 = torch.ops.aten.add.Tensor(floor, 2)
        ge_6 = torch.ops.aten.ge.Scalar(add_8, 0)
        lt_8 = torch.ops.aten.lt.Scalar(add_8, 456)
        ge_7 = torch.ops.aten.ge.Scalar(add_6, 0)
        lt_9 = torch.ops.aten.lt.Scalar(add_6, 345)
        logical_and_9 = torch.ops.aten.logical_and.default(ge_7, lt_9);  ge_7 = lt_9 = None
        logical_and_10 = torch.ops.aten.logical_and.default(lt_8, logical_and_9);  lt_8 = logical_and_9 = None
        logical_and_11 = torch.ops.aten.logical_and.default(ge_6, logical_and_10);  ge_6 = logical_and_10 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(add_8, torch.int64);  add_8 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(add_6, torch.int64);  add_6 = None
        full_default_13 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11 = torch.ops.aten.where.self(logical_and_11, convert_element_type_10, full_default_13);  convert_element_type_10 = full_default_13 = None
        full_default_14 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_12 = torch.ops.aten.where.self(logical_and_11, convert_element_type_11, full_default_14);  convert_element_type_11 = full_default_14 = None
        full_default_15 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16 = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13 = torch.ops.aten.where.self(logical_and_11, full_default_16, full_default_15);  logical_and_11 = full_default_16 = full_default_15 = None
        index_3 = torch.ops.aten.index.Tensor(arg0_1, [view_5, view_6, where_12, where_11]);  where_12 = where_11 = None
        mul_10 = torch.ops.aten.mul.Tensor(index_3, where_13);  index_3 = where_13 = None
        add_9 = torch.ops.aten.add.Tensor(sub_4, 1.0)
        mul_11 = torch.ops.aten.mul.Tensor(add_9, -0.75)
        sub_7 = torch.ops.aten.sub.Tensor(mul_11, -3.75);  mul_11 = None
        mul_12 = torch.ops.aten.mul.Tensor(sub_7, add_9);  sub_7 = None
        add_10 = torch.ops.aten.add.Tensor(mul_12, -6.0);  mul_12 = None
        mul_13 = torch.ops.aten.mul.Tensor(add_10, add_9);  add_10 = add_9 = None
        sub_8 = torch.ops.aten.sub.Tensor(mul_13, -3.0);  mul_13 = None
        mul_14 = torch.ops.aten.mul.Tensor(sub_4, 1.25)
        sub_9 = torch.ops.aten.sub.Tensor(mul_14, 2.25);  mul_14 = None
        mul_15 = torch.ops.aten.mul.Tensor(sub_9, sub_4);  sub_9 = None
        mul_16 = torch.ops.aten.mul.Tensor(mul_15, sub_4);  mul_15 = None
        add_11 = torch.ops.aten.add.Tensor(mul_16, 1);  mul_16 = None
        sub_10 = torch.ops.aten.sub.Tensor(1.0, sub_4)
        mul_17 = torch.ops.aten.mul.Tensor(sub_10, 1.25)
        sub_11 = torch.ops.aten.sub.Tensor(mul_17, 2.25);  mul_17 = None
        mul_18 = torch.ops.aten.mul.Tensor(sub_11, sub_10);  sub_11 = None
        mul_19 = torch.ops.aten.mul.Tensor(mul_18, sub_10);  mul_18 = sub_10 = None
        add_12 = torch.ops.aten.add.Tensor(mul_19, 1);  mul_19 = None
        sub_12 = torch.ops.aten.sub.Tensor(2.0, sub_4)
        mul_20 = torch.ops.aten.mul.Tensor(sub_12, -0.75)
        sub_13 = torch.ops.aten.sub.Tensor(mul_20, -3.75);  mul_20 = None
        mul_21 = torch.ops.aten.mul.Tensor(sub_13, sub_12);  sub_13 = None
        add_13 = torch.ops.aten.add.Tensor(mul_21, -6.0);  mul_21 = None
        mul_22 = torch.ops.aten.mul.Tensor(add_13, sub_12);  add_13 = sub_12 = None
        sub_14 = torch.ops.aten.sub.Tensor(mul_22, -3.0);  mul_22 = None
        mul_23 = torch.ops.aten.mul.Tensor(mul_7, sub_8);  mul_7 = sub_8 = None
        mul_24 = torch.ops.aten.mul.Tensor(mul_8, add_11);  mul_8 = add_11 = None
        add_14 = torch.ops.aten.add.Tensor(mul_23, mul_24);  mul_23 = mul_24 = None
        mul_25 = torch.ops.aten.mul.Tensor(mul_9, add_12);  mul_9 = add_12 = None
        add_15 = torch.ops.aten.add.Tensor(add_14, mul_25);  add_14 = mul_25 = None
        mul_26 = torch.ops.aten.mul.Tensor(mul_10, sub_14);  mul_10 = sub_14 = None
        add_16 = torch.ops.aten.add.Tensor(add_15, mul_26);  add_15 = mul_26 = None
        add_17 = torch.ops.aten.add.Tensor(floor_1, 0)
        sub_15 = torch.ops.aten.sub.Tensor(floor, 1)
        ge_8 = torch.ops.aten.ge.Scalar(sub_15, 0)
        lt_10 = torch.ops.aten.lt.Scalar(sub_15, 456)
        ge_9 = torch.ops.aten.ge.Scalar(add_17, 0)
        lt_11 = torch.ops.aten.lt.Scalar(add_17, 345)
        logical_and_12 = torch.ops.aten.logical_and.default(ge_9, lt_11);  ge_9 = lt_11 = None
        logical_and_13 = torch.ops.aten.logical_and.default(lt_10, logical_and_12);  lt_10 = logical_and_12 = None
        logical_and_14 = torch.ops.aten.logical_and.default(ge_8, logical_and_13);  ge_8 = logical_and_13 = None
        convert_element_type_12 = torch.ops.prims.convert_element_type.default(sub_15, torch.int64);  sub_15 = None
        convert_element_type_13 = torch.ops.prims.convert_element_type.default(add_17, torch.int64)
        full_default_17 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_14 = torch.ops.aten.where.self(logical_and_14, convert_element_type_12, full_default_17);  convert_element_type_12 = full_default_17 = None
        full_default_18 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_15 = torch.ops.aten.where.self(logical_and_14, convert_element_type_13, full_default_18);  convert_element_type_13 = full_default_18 = None
        full_default_19 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20 = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16 = torch.ops.aten.where.self(logical_and_14, full_default_20, full_default_19);  logical_and_14 = full_default_20 = full_default_19 = None
        index_4 = torch.ops.aten.index.Tensor(arg0_1, [view_5, view_6, where_15, where_14]);  where_15 = where_14 = None
        mul_27 = torch.ops.aten.mul.Tensor(index_4, where_16);  index_4 = where_16 = None
        ge_10 = torch.ops.aten.ge.Scalar(floor, 0)
        lt_12 = torch.ops.aten.lt.Scalar(floor, 456)
        ge_11 = torch.ops.aten.ge.Scalar(add_17, 0)
        lt_13 = torch.ops.aten.lt.Scalar(add_17, 345)
        logical_and_15 = torch.ops.aten.logical_and.default(ge_11, lt_13);  ge_11 = lt_13 = None
        logical_and_16 = torch.ops.aten.logical_and.default(lt_12, logical_and_15);  lt_12 = logical_and_15 = None
        logical_and_17 = torch.ops.aten.logical_and.default(ge_10, logical_and_16);  ge_10 = logical_and_16 = None
        convert_element_type_14 = torch.ops.prims.convert_element_type.default(floor, torch.int64)
        convert_element_type_15 = torch.ops.prims.convert_element_type.default(add_17, torch.int64)
        full_default_21 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_17 = torch.ops.aten.where.self(logical_and_17, convert_element_type_14, full_default_21);  convert_element_type_14 = full_default_21 = None
        full_default_22 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_18 = torch.ops.aten.where.self(logical_and_17, convert_element_type_15, full_default_22);  convert_element_type_15 = full_default_22 = None
        full_default_23 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24 = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_19 = torch.ops.aten.where.self(logical_and_17, full_default_24, full_default_23);  logical_and_17 = full_default_24 = full_default_23 = None
        index_5 = torch.ops.aten.index.Tensor(arg0_1, [view_5, view_6, where_18, where_17]);  where_18 = where_17 = None
        mul_28 = torch.ops.aten.mul.Tensor(index_5, where_19);  index_5 = where_19 = None
        add_18 = torch.ops.aten.add.Tensor(floor, 1)
        ge_12 = torch.ops.aten.ge.Scalar(add_18, 0)
        lt_14 = torch.ops.aten.lt.Scalar(add_18, 456)
        ge_13 = torch.ops.aten.ge.Scalar(add_17, 0)
        lt_15 = torch.ops.aten.lt.Scalar(add_17, 345)
        logical_and_18 = torch.ops.aten.logical_and.default(ge_13, lt_15);  ge_13 = lt_15 = None
        logical_and_19 = torch.ops.aten.logical_and.default(lt_14, logical_and_18);  lt_14 = logical_and_18 = None
        logical_and_20 = torch.ops.aten.logical_and.default(ge_12, logical_and_19);  ge_12 = logical_and_19 = None
        convert_element_type_16 = torch.ops.prims.convert_element_type.default(add_18, torch.int64);  add_18 = None
        convert_element_type_17 = torch.ops.prims.convert_element_type.default(add_17, torch.int64)
        full_default_25 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_20 = torch.ops.aten.where.self(logical_and_20, convert_element_type_16, full_default_25);  convert_element_type_16 = full_default_25 = None
        full_default_26 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_21 = torch.ops.aten.where.self(logical_and_20, convert_element_type_17, full_default_26);  convert_element_type_17 = full_default_26 = None
        full_default_27 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_28 = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_22 = torch.ops.aten.where.self(logical_and_20, full_default_28, full_default_27);  logical_and_20 = full_default_28 = full_default_27 = None
        index_6 = torch.ops.aten.index.Tensor(arg0_1, [view_5, view_6, where_21, where_20]);  where_21 = where_20 = None
        mul_29 = torch.ops.aten.mul.Tensor(index_6, where_22);  index_6 = where_22 = None
        add_19 = torch.ops.aten.add.Tensor(floor, 2)
        ge_14 = torch.ops.aten.ge.Scalar(add_19, 0)
        lt_16 = torch.ops.aten.lt.Scalar(add_19, 456)
        ge_15 = torch.ops.aten.ge.Scalar(add_17, 0)
        lt_17 = torch.ops.aten.lt.Scalar(add_17, 345)
        logical_and_21 = torch.ops.aten.logical_and.default(ge_15, lt_17);  ge_15 = lt_17 = None
        logical_and_22 = torch.ops.aten.logical_and.default(lt_16, logical_and_21);  lt_16 = logical_and_21 = None
        logical_and_23 = torch.ops.aten.logical_and.default(ge_14, logical_and_22);  ge_14 = logical_and_22 = None
        convert_element_type_18 = torch.ops.prims.convert_element_type.default(add_19, torch.int64);  add_19 = None
        convert_element_type_19 = torch.ops.prims.convert_element_type.default(add_17, torch.int64);  add_17 = None
        full_default_29 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_23 = torch.ops.aten.where.self(logical_and_23, convert_element_type_18, full_default_29);  convert_element_type_18 = full_default_29 = None
        full_default_30 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24 = torch.ops.aten.where.self(logical_and_23, convert_element_type_19, full_default_30);  convert_element_type_19 = full_default_30 = None
        full_default_31 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_32 = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25 = torch.ops.aten.where.self(logical_and_23, full_default_32, full_default_31);  logical_and_23 = full_default_32 = full_default_31 = None
        index_7 = torch.ops.aten.index.Tensor(arg0_1, [view_5, view_6, where_24, where_23]);  where_24 = where_23 = None
        mul_30 = torch.ops.aten.mul.Tensor(index_7, where_25);  index_7 = where_25 = None
        add_20 = torch.ops.aten.add.Tensor(sub_4, 1.0)
        mul_31 = torch.ops.aten.mul.Tensor(add_20, -0.75)
        sub_16 = torch.ops.aten.sub.Tensor(mul_31, -3.75);  mul_31 = None
        mul_32 = torch.ops.aten.mul.Tensor(sub_16, add_20);  sub_16 = None
        add_21 = torch.ops.aten.add.Tensor(mul_32, -6.0);  mul_32 = None
        mul_33 = torch.ops.aten.mul.Tensor(add_21, add_20);  add_21 = add_20 = None
        sub_17 = torch.ops.aten.sub.Tensor(mul_33, -3.0);  mul_33 = None
        mul_34 = torch.ops.aten.mul.Tensor(sub_4, 1.25)
        sub_18 = torch.ops.aten.sub.Tensor(mul_34, 2.25);  mul_34 = None
        mul_35 = torch.ops.aten.mul.Tensor(sub_18, sub_4);  sub_18 = None
        mul_36 = torch.ops.aten.mul.Tensor(mul_35, sub_4);  mul_35 = None
        add_22 = torch.ops.aten.add.Tensor(mul_36, 1);  mul_36 = None
        sub_19 = torch.ops.aten.sub.Tensor(1.0, sub_4)
        mul_37 = torch.ops.aten.mul.Tensor(sub_19, 1.25)
        sub_20 = torch.ops.aten.sub.Tensor(mul_37, 2.25);  mul_37 = None
        mul_38 = torch.ops.aten.mul.Tensor(sub_20, sub_19);  sub_20 = None
        mul_39 = torch.ops.aten.mul.Tensor(mul_38, sub_19);  mul_38 = sub_19 = None
        add_23 = torch.ops.aten.add.Tensor(mul_39, 1);  mul_39 = None
        sub_21 = torch.ops.aten.sub.Tensor(2.0, sub_4)
        mul_40 = torch.ops.aten.mul.Tensor(sub_21, -0.75)
        sub_22 = torch.ops.aten.sub.Tensor(mul_40, -3.75);  mul_40 = None
        mul_41 = torch.ops.aten.mul.Tensor(sub_22, sub_21);  sub_22 = None
        add_24 = torch.ops.aten.add.Tensor(mul_41, -6.0);  mul_41 = None
        mul_42 = torch.ops.aten.mul.Tensor(add_24, sub_21);  add_24 = sub_21 = None
        sub_23 = torch.ops.aten.sub.Tensor(mul_42, -3.0);  mul_42 = None
        mul_43 = torch.ops.aten.mul.Tensor(mul_27, sub_17);  mul_27 = sub_17 = None
        mul_44 = torch.ops.aten.mul.Tensor(mul_28, add_22);  mul_28 = add_22 = None
        add_25 = torch.ops.aten.add.Tensor(mul_43, mul_44);  mul_43 = mul_44 = None
        mul_45 = torch.ops.aten.mul.Tensor(mul_29, add_23);  mul_29 = add_23 = None
        add_26 = torch.ops.aten.add.Tensor(add_25, mul_45);  add_25 = mul_45 = None
        mul_46 = torch.ops.aten.mul.Tensor(mul_30, sub_23);  mul_30 = sub_23 = None
        add_27 = torch.ops.aten.add.Tensor(add_26, mul_46);  add_26 = mul_46 = None
        add_28 = torch.ops.aten.add.Tensor(floor_1, 1)
        sub_24 = torch.ops.aten.sub.Tensor(floor, 1)
        ge_16 = torch.ops.aten.ge.Scalar(sub_24, 0)
        lt_18 = torch.ops.aten.lt.Scalar(sub_24, 456)
        ge_17 = torch.ops.aten.ge.Scalar(add_28, 0)
        lt_19 = torch.ops.aten.lt.Scalar(add_28, 345)
        logical_and_24 = torch.ops.aten.logical_and.default(ge_17, lt_19);  ge_17 = lt_19 = None
        logical_and_25 = torch.ops.aten.logical_and.default(lt_18, logical_and_24);  lt_18 = logical_and_24 = None
        logical_and_26 = torch.ops.aten.logical_and.default(ge_16, logical_and_25);  ge_16 = logical_and_25 = None
        convert_element_type_20 = torch.ops.prims.convert_element_type.default(sub_24, torch.int64);  sub_24 = None
        convert_element_type_21 = torch.ops.prims.convert_element_type.default(add_28, torch.int64)
        full_default_33 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26 = torch.ops.aten.where.self(logical_and_26, convert_element_type_20, full_default_33);  convert_element_type_20 = full_default_33 = None
        full_default_34 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_27 = torch.ops.aten.where.self(logical_and_26, convert_element_type_21, full_default_34);  convert_element_type_21 = full_default_34 = None
        full_default_35 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_36 = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_28 = torch.ops.aten.where.self(logical_and_26, full_default_36, full_default_35);  logical_and_26 = full_default_36 = full_default_35 = None
        index_8 = torch.ops.aten.index.Tensor(arg0_1, [view_5, view_6, where_27, where_26]);  where_27 = where_26 = None
        mul_47 = torch.ops.aten.mul.Tensor(index_8, where_28);  index_8 = where_28 = None
        ge_18 = torch.ops.aten.ge.Scalar(floor, 0)
        lt_20 = torch.ops.aten.lt.Scalar(floor, 456)
        ge_19 = torch.ops.aten.ge.Scalar(add_28, 0)
        lt_21 = torch.ops.aten.lt.Scalar(add_28, 345)
        logical_and_27 = torch.ops.aten.logical_and.default(ge_19, lt_21);  ge_19 = lt_21 = None
        logical_and_28 = torch.ops.aten.logical_and.default(lt_20, logical_and_27);  lt_20 = logical_and_27 = None
        logical_and_29 = torch.ops.aten.logical_and.default(ge_18, logical_and_28);  ge_18 = logical_and_28 = None
        convert_element_type_22 = torch.ops.prims.convert_element_type.default(floor, torch.int64)
        convert_element_type_23 = torch.ops.prims.convert_element_type.default(add_28, torch.int64)
        full_default_37 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_29 = torch.ops.aten.where.self(logical_and_29, convert_element_type_22, full_default_37);  convert_element_type_22 = full_default_37 = None
        full_default_38 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_30 = torch.ops.aten.where.self(logical_and_29, convert_element_type_23, full_default_38);  convert_element_type_23 = full_default_38 = None
        full_default_39 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_40 = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_31 = torch.ops.aten.where.self(logical_and_29, full_default_40, full_default_39);  logical_and_29 = full_default_40 = full_default_39 = None
        index_9 = torch.ops.aten.index.Tensor(arg0_1, [view_5, view_6, where_30, where_29]);  where_30 = where_29 = None
        mul_48 = torch.ops.aten.mul.Tensor(index_9, where_31);  index_9 = where_31 = None
        add_29 = torch.ops.aten.add.Tensor(floor, 1)
        ge_20 = torch.ops.aten.ge.Scalar(add_29, 0)
        lt_22 = torch.ops.aten.lt.Scalar(add_29, 456)
        ge_21 = torch.ops.aten.ge.Scalar(add_28, 0)
        lt_23 = torch.ops.aten.lt.Scalar(add_28, 345)
        logical_and_30 = torch.ops.aten.logical_and.default(ge_21, lt_23);  ge_21 = lt_23 = None
        logical_and_31 = torch.ops.aten.logical_and.default(lt_22, logical_and_30);  lt_22 = logical_and_30 = None
        logical_and_32 = torch.ops.aten.logical_and.default(ge_20, logical_and_31);  ge_20 = logical_and_31 = None
        convert_element_type_24 = torch.ops.prims.convert_element_type.default(add_29, torch.int64);  add_29 = None
        convert_element_type_25 = torch.ops.prims.convert_element_type.default(add_28, torch.int64)
        full_default_41 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_32 = torch.ops.aten.where.self(logical_and_32, convert_element_type_24, full_default_41);  convert_element_type_24 = full_default_41 = None
        full_default_42 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_33 = torch.ops.aten.where.self(logical_and_32, convert_element_type_25, full_default_42);  convert_element_type_25 = full_default_42 = None
        full_default_43 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_44 = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_34 = torch.ops.aten.where.self(logical_and_32, full_default_44, full_default_43);  logical_and_32 = full_default_44 = full_default_43 = None
        index_10 = torch.ops.aten.index.Tensor(arg0_1, [view_5, view_6, where_33, where_32]);  where_33 = where_32 = None
        mul_49 = torch.ops.aten.mul.Tensor(index_10, where_34);  index_10 = where_34 = None
        add_30 = torch.ops.aten.add.Tensor(floor, 2)
        ge_22 = torch.ops.aten.ge.Scalar(add_30, 0)
        lt_24 = torch.ops.aten.lt.Scalar(add_30, 456)
        ge_23 = torch.ops.aten.ge.Scalar(add_28, 0)
        lt_25 = torch.ops.aten.lt.Scalar(add_28, 345)
        logical_and_33 = torch.ops.aten.logical_and.default(ge_23, lt_25);  ge_23 = lt_25 = None
        logical_and_34 = torch.ops.aten.logical_and.default(lt_24, logical_and_33);  lt_24 = logical_and_33 = None
        logical_and_35 = torch.ops.aten.logical_and.default(ge_22, logical_and_34);  ge_22 = logical_and_34 = None
        convert_element_type_26 = torch.ops.prims.convert_element_type.default(add_30, torch.int64);  add_30 = None
        convert_element_type_27 = torch.ops.prims.convert_element_type.default(add_28, torch.int64);  add_28 = None
        full_default_45 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_35 = torch.ops.aten.where.self(logical_and_35, convert_element_type_26, full_default_45);  convert_element_type_26 = full_default_45 = None
        full_default_46 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_36 = torch.ops.aten.where.self(logical_and_35, convert_element_type_27, full_default_46);  convert_element_type_27 = full_default_46 = None
        full_default_47 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_48 = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_37 = torch.ops.aten.where.self(logical_and_35, full_default_48, full_default_47);  logical_and_35 = full_default_48 = full_default_47 = None
        index_11 = torch.ops.aten.index.Tensor(arg0_1, [view_5, view_6, where_36, where_35]);  where_36 = where_35 = None
        mul_50 = torch.ops.aten.mul.Tensor(index_11, where_37);  index_11 = where_37 = None
        add_31 = torch.ops.aten.add.Tensor(sub_4, 1.0)
        mul_51 = torch.ops.aten.mul.Tensor(add_31, -0.75)
        sub_25 = torch.ops.aten.sub.Tensor(mul_51, -3.75);  mul_51 = None
        mul_52 = torch.ops.aten.mul.Tensor(sub_25, add_31);  sub_25 = None
        add_32 = torch.ops.aten.add.Tensor(mul_52, -6.0);  mul_52 = None
        mul_53 = torch.ops.aten.mul.Tensor(add_32, add_31);  add_32 = add_31 = None
        sub_26 = torch.ops.aten.sub.Tensor(mul_53, -3.0);  mul_53 = None
        mul_54 = torch.ops.aten.mul.Tensor(sub_4, 1.25)
        sub_27 = torch.ops.aten.sub.Tensor(mul_54, 2.25);  mul_54 = None
        mul_55 = torch.ops.aten.mul.Tensor(sub_27, sub_4);  sub_27 = None
        mul_56 = torch.ops.aten.mul.Tensor(mul_55, sub_4);  mul_55 = None
        add_33 = torch.ops.aten.add.Tensor(mul_56, 1);  mul_56 = None
        sub_28 = torch.ops.aten.sub.Tensor(1.0, sub_4)
        mul_57 = torch.ops.aten.mul.Tensor(sub_28, 1.25)
        sub_29 = torch.ops.aten.sub.Tensor(mul_57, 2.25);  mul_57 = None
        mul_58 = torch.ops.aten.mul.Tensor(sub_29, sub_28);  sub_29 = None
        mul_59 = torch.ops.aten.mul.Tensor(mul_58, sub_28);  mul_58 = sub_28 = None
        add_34 = torch.ops.aten.add.Tensor(mul_59, 1);  mul_59 = None
        sub_30 = torch.ops.aten.sub.Tensor(2.0, sub_4)
        mul_60 = torch.ops.aten.mul.Tensor(sub_30, -0.75)
        sub_31 = torch.ops.aten.sub.Tensor(mul_60, -3.75);  mul_60 = None
        mul_61 = torch.ops.aten.mul.Tensor(sub_31, sub_30);  sub_31 = None
        add_35 = torch.ops.aten.add.Tensor(mul_61, -6.0);  mul_61 = None
        mul_62 = torch.ops.aten.mul.Tensor(add_35, sub_30);  add_35 = sub_30 = None
        sub_32 = torch.ops.aten.sub.Tensor(mul_62, -3.0);  mul_62 = None
        mul_63 = torch.ops.aten.mul.Tensor(mul_47, sub_26);  mul_47 = sub_26 = None
        mul_64 = torch.ops.aten.mul.Tensor(mul_48, add_33);  mul_48 = add_33 = None
        add_36 = torch.ops.aten.add.Tensor(mul_63, mul_64);  mul_63 = mul_64 = None
        mul_65 = torch.ops.aten.mul.Tensor(mul_49, add_34);  mul_49 = add_34 = None
        add_37 = torch.ops.aten.add.Tensor(add_36, mul_65);  add_36 = mul_65 = None
        mul_66 = torch.ops.aten.mul.Tensor(mul_50, sub_32);  mul_50 = sub_32 = None
        add_38 = torch.ops.aten.add.Tensor(add_37, mul_66);  add_37 = mul_66 = None
        add_39 = torch.ops.aten.add.Tensor(floor_1, 2);  floor_1 = None
        sub_33 = torch.ops.aten.sub.Tensor(floor, 1)
        ge_24 = torch.ops.aten.ge.Scalar(sub_33, 0)
        lt_26 = torch.ops.aten.lt.Scalar(sub_33, 456)
        ge_25 = torch.ops.aten.ge.Scalar(add_39, 0)
        lt_27 = torch.ops.aten.lt.Scalar(add_39, 345)
        logical_and_36 = torch.ops.aten.logical_and.default(ge_25, lt_27);  ge_25 = lt_27 = None
        logical_and_37 = torch.ops.aten.logical_and.default(lt_26, logical_and_36);  lt_26 = logical_and_36 = None
        logical_and_38 = torch.ops.aten.logical_and.default(ge_24, logical_and_37);  ge_24 = logical_and_37 = None
        convert_element_type_28 = torch.ops.prims.convert_element_type.default(sub_33, torch.int64);  sub_33 = None
        convert_element_type_29 = torch.ops.prims.convert_element_type.default(add_39, torch.int64)
        full_default_49 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_38 = torch.ops.aten.where.self(logical_and_38, convert_element_type_28, full_default_49);  convert_element_type_28 = full_default_49 = None
        full_default_50 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_39 = torch.ops.aten.where.self(logical_and_38, convert_element_type_29, full_default_50);  convert_element_type_29 = full_default_50 = None
        full_default_51 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_52 = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_40 = torch.ops.aten.where.self(logical_and_38, full_default_52, full_default_51);  logical_and_38 = full_default_52 = full_default_51 = None
        index_12 = torch.ops.aten.index.Tensor(arg0_1, [view_5, view_6, where_39, where_38]);  where_39 = where_38 = None
        mul_67 = torch.ops.aten.mul.Tensor(index_12, where_40);  index_12 = where_40 = None
        ge_26 = torch.ops.aten.ge.Scalar(floor, 0)
        lt_28 = torch.ops.aten.lt.Scalar(floor, 456)
        ge_27 = torch.ops.aten.ge.Scalar(add_39, 0)
        lt_29 = torch.ops.aten.lt.Scalar(add_39, 345)
        logical_and_39 = torch.ops.aten.logical_and.default(ge_27, lt_29);  ge_27 = lt_29 = None
        logical_and_40 = torch.ops.aten.logical_and.default(lt_28, logical_and_39);  lt_28 = logical_and_39 = None
        logical_and_41 = torch.ops.aten.logical_and.default(ge_26, logical_and_40);  ge_26 = logical_and_40 = None
        convert_element_type_30 = torch.ops.prims.convert_element_type.default(floor, torch.int64)
        convert_element_type_31 = torch.ops.prims.convert_element_type.default(add_39, torch.int64)
        full_default_53 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_41 = torch.ops.aten.where.self(logical_and_41, convert_element_type_30, full_default_53);  convert_element_type_30 = full_default_53 = None
        full_default_54 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_42 = torch.ops.aten.where.self(logical_and_41, convert_element_type_31, full_default_54);  convert_element_type_31 = full_default_54 = None
        full_default_55 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_56 = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_43 = torch.ops.aten.where.self(logical_and_41, full_default_56, full_default_55);  logical_and_41 = full_default_56 = full_default_55 = None
        index_13 = torch.ops.aten.index.Tensor(arg0_1, [view_5, view_6, where_42, where_41]);  where_42 = where_41 = None
        mul_68 = torch.ops.aten.mul.Tensor(index_13, where_43);  index_13 = where_43 = None
        add_40 = torch.ops.aten.add.Tensor(floor, 1)
        ge_28 = torch.ops.aten.ge.Scalar(add_40, 0)
        lt_30 = torch.ops.aten.lt.Scalar(add_40, 456)
        ge_29 = torch.ops.aten.ge.Scalar(add_39, 0)
        lt_31 = torch.ops.aten.lt.Scalar(add_39, 345)
        logical_and_42 = torch.ops.aten.logical_and.default(ge_29, lt_31);  ge_29 = lt_31 = None
        logical_and_43 = torch.ops.aten.logical_and.default(lt_30, logical_and_42);  lt_30 = logical_and_42 = None
        logical_and_44 = torch.ops.aten.logical_and.default(ge_28, logical_and_43);  ge_28 = logical_and_43 = None
        convert_element_type_32 = torch.ops.prims.convert_element_type.default(add_40, torch.int64);  add_40 = None
        convert_element_type_33 = torch.ops.prims.convert_element_type.default(add_39, torch.int64)
        full_default_57 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_44 = torch.ops.aten.where.self(logical_and_44, convert_element_type_32, full_default_57);  convert_element_type_32 = full_default_57 = None
        full_default_58 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_45 = torch.ops.aten.where.self(logical_and_44, convert_element_type_33, full_default_58);  convert_element_type_33 = full_default_58 = None
        full_default_59 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_60 = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_46 = torch.ops.aten.where.self(logical_and_44, full_default_60, full_default_59);  logical_and_44 = full_default_60 = full_default_59 = None
        index_14 = torch.ops.aten.index.Tensor(arg0_1, [view_5, view_6, where_45, where_44]);  where_45 = where_44 = None
        mul_69 = torch.ops.aten.mul.Tensor(index_14, where_46);  index_14 = where_46 = None
        add_41 = torch.ops.aten.add.Tensor(floor, 2);  floor = None
        ge_30 = torch.ops.aten.ge.Scalar(add_41, 0)
        lt_32 = torch.ops.aten.lt.Scalar(add_41, 456)
        ge_31 = torch.ops.aten.ge.Scalar(add_39, 0)
        lt_33 = torch.ops.aten.lt.Scalar(add_39, 345)
        logical_and_45 = torch.ops.aten.logical_and.default(ge_31, lt_33);  ge_31 = lt_33 = None
        logical_and_46 = torch.ops.aten.logical_and.default(lt_32, logical_and_45);  lt_32 = logical_and_45 = None
        logical_and_47 = torch.ops.aten.logical_and.default(ge_30, logical_and_46);  ge_30 = logical_and_46 = None
        convert_element_type_34 = torch.ops.prims.convert_element_type.default(add_41, torch.int64);  add_41 = None
        convert_element_type_35 = torch.ops.prims.convert_element_type.default(add_39, torch.int64);  add_39 = None
        full_default_61 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_47 = torch.ops.aten.where.self(logical_and_47, convert_element_type_34, full_default_61);  convert_element_type_34 = full_default_61 = None
        full_default_62 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_48 = torch.ops.aten.where.self(logical_and_47, convert_element_type_35, full_default_62);  convert_element_type_35 = full_default_62 = None
        full_default_63 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_64 = torch.ops.aten.full.default([], 1, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_49 = torch.ops.aten.where.self(logical_and_47, full_default_64, full_default_63);  logical_and_47 = full_default_64 = full_default_63 = None
        index_15 = torch.ops.aten.index.Tensor(arg0_1, [view_5, view_6, where_48, where_47]);  arg0_1 = view_5 = view_6 = where_48 = where_47 = None
        mul_70 = torch.ops.aten.mul.Tensor(index_15, where_49);  index_15 = where_49 = None
        add_42 = torch.ops.aten.add.Tensor(sub_4, 1.0)
        mul_71 = torch.ops.aten.mul.Tensor(add_42, -0.75)
        sub_34 = torch.ops.aten.sub.Tensor(mul_71, -3.75);  mul_71 = None
        mul_72 = torch.ops.aten.mul.Tensor(sub_34, add_42);  sub_34 = None
        add_43 = torch.ops.aten.add.Tensor(mul_72, -6.0);  mul_72 = None
        mul_73 = torch.ops.aten.mul.Tensor(add_43, add_42);  add_43 = add_42 = None
        sub_35 = torch.ops.aten.sub.Tensor(mul_73, -3.0);  mul_73 = None
        mul_74 = torch.ops.aten.mul.Tensor(sub_4, 1.25)
        sub_36 = torch.ops.aten.sub.Tensor(mul_74, 2.25);  mul_74 = None
        mul_75 = torch.ops.aten.mul.Tensor(sub_36, sub_4);  sub_36 = None
        mul_76 = torch.ops.aten.mul.Tensor(mul_75, sub_4);  mul_75 = None
        add_44 = torch.ops.aten.add.Tensor(mul_76, 1);  mul_76 = None
        sub_37 = torch.ops.aten.sub.Tensor(1.0, sub_4)
        mul_77 = torch.ops.aten.mul.Tensor(sub_37, 1.25)
        sub_38 = torch.ops.aten.sub.Tensor(mul_77, 2.25);  mul_77 = None
        mul_78 = torch.ops.aten.mul.Tensor(sub_38, sub_37);  sub_38 = None
        mul_79 = torch.ops.aten.mul.Tensor(mul_78, sub_37);  mul_78 = sub_37 = None
        add_45 = torch.ops.aten.add.Tensor(mul_79, 1);  mul_79 = None
        sub_39 = torch.ops.aten.sub.Tensor(2.0, sub_4);  sub_4 = None
        mul_80 = torch.ops.aten.mul.Tensor(sub_39, -0.75)
        sub_40 = torch.ops.aten.sub.Tensor(mul_80, -3.75);  mul_80 = None
        mul_81 = torch.ops.aten.mul.Tensor(sub_40, sub_39);  sub_40 = None
        add_46 = torch.ops.aten.add.Tensor(mul_81, -6.0);  mul_81 = None
        mul_82 = torch.ops.aten.mul.Tensor(add_46, sub_39);  add_46 = sub_39 = None
        sub_41 = torch.ops.aten.sub.Tensor(mul_82, -3.0);  mul_82 = None
        mul_83 = torch.ops.aten.mul.Tensor(mul_67, sub_35);  mul_67 = sub_35 = None
        mul_84 = torch.ops.aten.mul.Tensor(mul_68, add_44);  mul_68 = add_44 = None
        add_47 = torch.ops.aten.add.Tensor(mul_83, mul_84);  mul_83 = mul_84 = None
        mul_85 = torch.ops.aten.mul.Tensor(mul_69, add_45);  mul_69 = add_45 = None
        add_48 = torch.ops.aten.add.Tensor(add_47, mul_85);  add_47 = mul_85 = None
        mul_86 = torch.ops.aten.mul.Tensor(mul_70, sub_41);  mul_70 = sub_41 = None
        add_49 = torch.ops.aten.add.Tensor(add_48, mul_86);  add_48 = mul_86 = None
        add_50 = torch.ops.aten.add.Tensor(sub_5, 1.0)
        mul_87 = torch.ops.aten.mul.Tensor(add_50, -0.75)
        sub_42 = torch.ops.aten.sub.Tensor(mul_87, -3.75);  mul_87 = None
        mul_88 = torch.ops.aten.mul.Tensor(sub_42, add_50);  sub_42 = None
        add_51 = torch.ops.aten.add.Tensor(mul_88, -6.0);  mul_88 = None
        mul_89 = torch.ops.aten.mul.Tensor(add_51, add_50);  add_51 = add_50 = None
        sub_43 = torch.ops.aten.sub.Tensor(mul_89, -3.0);  mul_89 = None
        mul_90 = torch.ops.aten.mul.Tensor(sub_5, 1.25)
        sub_44 = torch.ops.aten.sub.Tensor(mul_90, 2.25);  mul_90 = None
        mul_91 = torch.ops.aten.mul.Tensor(sub_44, sub_5);  sub_44 = None
        mul_92 = torch.ops.aten.mul.Tensor(mul_91, sub_5);  mul_91 = None
        add_52 = torch.ops.aten.add.Tensor(mul_92, 1);  mul_92 = None
        sub_45 = torch.ops.aten.sub.Tensor(1.0, sub_5)
        mul_93 = torch.ops.aten.mul.Tensor(sub_45, 1.25)
        sub_46 = torch.ops.aten.sub.Tensor(mul_93, 2.25);  mul_93 = None
        mul_94 = torch.ops.aten.mul.Tensor(sub_46, sub_45);  sub_46 = None
        mul_95 = torch.ops.aten.mul.Tensor(mul_94, sub_45);  mul_94 = sub_45 = None
        add_53 = torch.ops.aten.add.Tensor(mul_95, 1);  mul_95 = None
        sub_47 = torch.ops.aten.sub.Tensor(2.0, sub_5);  sub_5 = None
        mul_96 = torch.ops.aten.mul.Tensor(sub_47, -0.75)
        sub_48 = torch.ops.aten.sub.Tensor(mul_96, -3.75);  mul_96 = None
        mul_97 = torch.ops.aten.mul.Tensor(sub_48, sub_47);  sub_48 = None
        add_54 = torch.ops.aten.add.Tensor(mul_97, -6.0);  mul_97 = None
        mul_98 = torch.ops.aten.mul.Tensor(add_54, sub_47);  add_54 = sub_47 = None
        sub_49 = torch.ops.aten.sub.Tensor(mul_98, -3.0);  mul_98 = None
        mul_99 = torch.ops.aten.mul.Tensor(add_16, sub_43);  add_16 = sub_43 = None
        mul_100 = torch.ops.aten.mul.Tensor(add_27, add_52);  add_27 = add_52 = None
        add_55 = torch.ops.aten.add.Tensor(mul_99, mul_100);  mul_99 = mul_100 = None
        mul_101 = torch.ops.aten.mul.Tensor(add_38, add_53);  add_38 = add_53 = None
        add_56 = torch.ops.aten.add.Tensor(add_55, mul_101);  add_55 = mul_101 = None
        mul_102 = torch.ops.aten.mul.Tensor(add_49, sub_49);  add_49 = sub_49 = None
        add_57 = torch.ops.aten.add.Tensor(add_56, mul_102);  add_56 = mul_102 = None
        return (add_57,)
        
def load_args(reader):
    buf0 = reader.storage(None, 3775680, device=device(type='cuda', index=0))
    reader.tensor(buf0, (2, 3, 345, 456), (471960, 1, 1368, 3), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 48, device=device(type='cuda', index=0))
    reader.tensor(buf1, (2, 2, 3), is_leaf=True)  # arg1_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
