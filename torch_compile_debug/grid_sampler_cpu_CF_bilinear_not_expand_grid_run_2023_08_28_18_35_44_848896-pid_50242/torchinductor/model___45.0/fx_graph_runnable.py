
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



# torch version: 2.1.0a0+git52598e9
# torch cuda version: 12.1
# torch git version: 52598e95500417e2328246166b773c128feed100


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Mon_Apr__3_17:16:06_PDT_2023 
# Cuda compilation tools, release 12.1, V12.1.105 
# Build cuda_12.1.r12.1/compiler.32688072_0 

# GPU Hardware Info: 
# NVIDIA GeForce RTX 4090 : 1 
# NVIDIA GeForce GTX 1080 Ti : 1 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1):
        iota = torch.ops.prims.iota.default(456, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
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
        iota_1 = torch.ops.prims.iota.default(345, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
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
        full_default = torch.ops.aten.full.default([1, 1, 1], 1, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
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
        view_3 = torch.ops.aten.view.default(sum_1, [8, 345, 456, 2]);  sum_1 = None
        iota_2 = torch.ops.prims.iota.default(8, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_4 = torch.ops.aten.view.default(iota_2, [8, 1, 1, 1]);  iota_2 = None
        iota_3 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_5 = torch.ops.aten.view.default(iota_3, [1, 3, 1, 1]);  iota_3 = None
        select = torch.ops.aten.select.int(view_3, 3, 0)
        select_1 = torch.ops.aten.select.int(view_3, 3, 1);  view_3 = None
        mul_5 = torch.ops.aten.mul.Tensor(select, 228.0);  select = None
        add_4 = torch.ops.aten.add.Tensor(mul_5, 227.5);  mul_5 = None
        mul_6 = torch.ops.aten.mul.Tensor(select_1, 172.5);  select_1 = None
        add_5 = torch.ops.aten.add.Tensor(mul_6, 172.0);  mul_6 = None
        floor = torch.ops.aten.floor.default(add_4)
        floor_1 = torch.ops.aten.floor.default(add_5)
        add_6 = torch.ops.aten.add.Tensor(floor, 1)
        add_7 = torch.ops.aten.add.Tensor(floor_1, 1)
        sub_4 = torch.ops.aten.sub.Tensor(add_6, add_4)
        sub_5 = torch.ops.aten.sub.Tensor(add_7, add_5)
        mul_7 = torch.ops.aten.mul.Tensor(sub_4, sub_5);  sub_4 = sub_5 = None
        sub_6 = torch.ops.aten.sub.Tensor(add_4, floor)
        sub_7 = torch.ops.aten.sub.Tensor(add_7, add_5)
        mul_8 = torch.ops.aten.mul.Tensor(sub_6, sub_7);  sub_6 = sub_7 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_6, add_4)
        sub_9 = torch.ops.aten.sub.Tensor(add_5, floor_1)
        mul_9 = torch.ops.aten.mul.Tensor(sub_8, sub_9);  sub_8 = sub_9 = None
        sub_10 = torch.ops.aten.sub.Tensor(add_4, floor);  add_4 = None
        sub_11 = torch.ops.aten.sub.Tensor(add_5, floor_1);  add_5 = None
        mul_10 = torch.ops.aten.mul.Tensor(sub_10, sub_11);  sub_10 = sub_11 = None
        ge = torch.ops.aten.ge.Scalar(floor, 0)
        lt_2 = torch.ops.aten.lt.Scalar(floor, 456)
        ge_1 = torch.ops.aten.ge.Scalar(floor_1, 0)
        lt_3 = torch.ops.aten.lt.Scalar(floor_1, 345)
        logical_and = torch.ops.aten.logical_and.default(ge_1, lt_3);  ge_1 = lt_3 = None
        logical_and_1 = torch.ops.aten.logical_and.default(lt_2, logical_and);  lt_2 = logical_and = None
        logical_and_2 = torch.ops.aten.logical_and.default(ge, logical_and_1);  ge = logical_and_1 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(floor, torch.int64)
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(floor_1, torch.int64)
        full_default_1 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_2 = torch.ops.aten.where.self(logical_and_2, convert_element_type_4, full_default_1);  convert_element_type_4 = full_default_1 = None
        view_6 = torch.ops.aten.view.default(where_2, [8, 1, 345, 456]);  where_2 = None
        full_default_2 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_3 = torch.ops.aten.where.self(logical_and_2, convert_element_type_5, full_default_2);  convert_element_type_5 = full_default_2 = None
        view_7 = torch.ops.aten.view.default(where_3, [8, 1, 345, 456]);  where_3 = None
        full_default_3 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_4 = torch.ops.aten.where.self(logical_and_2, mul_7, full_default_3);  logical_and_2 = mul_7 = full_default_3 = None
        view_8 = torch.ops.aten.view.default(where_4, [8, 1, 345, 456]);  where_4 = None
        index = torch.ops.aten.index.Tensor(arg0_1, [view_4, view_5, view_7, view_6]);  view_7 = view_6 = None
        mul_11 = torch.ops.aten.mul.Tensor(index, view_8);  index = view_8 = None
        ge_2 = torch.ops.aten.ge.Scalar(add_6, 0)
        lt_4 = torch.ops.aten.lt.Scalar(add_6, 456)
        ge_3 = torch.ops.aten.ge.Scalar(floor_1, 0)
        lt_5 = torch.ops.aten.lt.Scalar(floor_1, 345)
        logical_and_3 = torch.ops.aten.logical_and.default(ge_3, lt_5);  ge_3 = lt_5 = None
        logical_and_4 = torch.ops.aten.logical_and.default(lt_4, logical_and_3);  lt_4 = logical_and_3 = None
        logical_and_5 = torch.ops.aten.logical_and.default(ge_2, logical_and_4);  ge_2 = logical_and_4 = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(add_6, torch.int64)
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(floor_1, torch.int64);  floor_1 = None
        full_default_4 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_5 = torch.ops.aten.where.self(logical_and_5, convert_element_type_6, full_default_4);  convert_element_type_6 = full_default_4 = None
        view_9 = torch.ops.aten.view.default(where_5, [8, 1, 345, 456]);  where_5 = None
        full_default_5 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_6 = torch.ops.aten.where.self(logical_and_5, convert_element_type_7, full_default_5);  convert_element_type_7 = full_default_5 = None
        view_10 = torch.ops.aten.view.default(where_6, [8, 1, 345, 456]);  where_6 = None
        full_default_6 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_7 = torch.ops.aten.where.self(logical_and_5, mul_8, full_default_6);  logical_and_5 = mul_8 = full_default_6 = None
        view_11 = torch.ops.aten.view.default(where_7, [8, 1, 345, 456]);  where_7 = None
        index_1 = torch.ops.aten.index.Tensor(arg0_1, [view_4, view_5, view_10, view_9]);  view_10 = view_9 = None
        mul_12 = torch.ops.aten.mul.Tensor(index_1, view_11);  index_1 = view_11 = None
        add_8 = torch.ops.aten.add.Tensor(mul_11, mul_12);  mul_11 = mul_12 = None
        ge_4 = torch.ops.aten.ge.Scalar(floor, 0)
        lt_6 = torch.ops.aten.lt.Scalar(floor, 456)
        ge_5 = torch.ops.aten.ge.Scalar(add_7, 0)
        lt_7 = torch.ops.aten.lt.Scalar(add_7, 345)
        logical_and_6 = torch.ops.aten.logical_and.default(ge_5, lt_7);  ge_5 = lt_7 = None
        logical_and_7 = torch.ops.aten.logical_and.default(lt_6, logical_and_6);  lt_6 = logical_and_6 = None
        logical_and_8 = torch.ops.aten.logical_and.default(ge_4, logical_and_7);  ge_4 = logical_and_7 = None
        convert_element_type_8 = torch.ops.prims.convert_element_type.default(floor, torch.int64);  floor = None
        convert_element_type_9 = torch.ops.prims.convert_element_type.default(add_7, torch.int64)
        full_default_7 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_8 = torch.ops.aten.where.self(logical_and_8, convert_element_type_8, full_default_7);  convert_element_type_8 = full_default_7 = None
        view_12 = torch.ops.aten.view.default(where_8, [8, 1, 345, 456]);  where_8 = None
        full_default_8 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_9 = torch.ops.aten.where.self(logical_and_8, convert_element_type_9, full_default_8);  convert_element_type_9 = full_default_8 = None
        view_13 = torch.ops.aten.view.default(where_9, [8, 1, 345, 456]);  where_9 = None
        full_default_9 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_10 = torch.ops.aten.where.self(logical_and_8, mul_9, full_default_9);  logical_and_8 = mul_9 = full_default_9 = None
        view_14 = torch.ops.aten.view.default(where_10, [8, 1, 345, 456]);  where_10 = None
        index_2 = torch.ops.aten.index.Tensor(arg0_1, [view_4, view_5, view_13, view_12]);  view_13 = view_12 = None
        mul_13 = torch.ops.aten.mul.Tensor(index_2, view_14);  index_2 = view_14 = None
        add_9 = torch.ops.aten.add.Tensor(add_8, mul_13);  add_8 = mul_13 = None
        ge_6 = torch.ops.aten.ge.Scalar(add_6, 0)
        lt_8 = torch.ops.aten.lt.Scalar(add_6, 456)
        ge_7 = torch.ops.aten.ge.Scalar(add_7, 0)
        lt_9 = torch.ops.aten.lt.Scalar(add_7, 345)
        logical_and_9 = torch.ops.aten.logical_and.default(ge_7, lt_9);  ge_7 = lt_9 = None
        logical_and_10 = torch.ops.aten.logical_and.default(lt_8, logical_and_9);  lt_8 = logical_and_9 = None
        logical_and_11 = torch.ops.aten.logical_and.default(ge_6, logical_and_10);  ge_6 = logical_and_10 = None
        convert_element_type_10 = torch.ops.prims.convert_element_type.default(add_6, torch.int64);  add_6 = None
        convert_element_type_11 = torch.ops.prims.convert_element_type.default(add_7, torch.int64);  add_7 = None
        full_default_10 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_11 = torch.ops.aten.where.self(logical_and_11, convert_element_type_10, full_default_10);  convert_element_type_10 = full_default_10 = None
        view_15 = torch.ops.aten.view.default(where_11, [8, 1, 345, 456]);  where_11 = None
        full_default_11 = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_12 = torch.ops.aten.where.self(logical_and_11, convert_element_type_11, full_default_11);  convert_element_type_11 = full_default_11 = None
        view_16 = torch.ops.aten.view.default(where_12, [8, 1, 345, 456]);  where_12 = None
        full_default_12 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_13 = torch.ops.aten.where.self(logical_and_11, mul_10, full_default_12);  logical_and_11 = mul_10 = full_default_12 = None
        view_17 = torch.ops.aten.view.default(where_13, [8, 1, 345, 456]);  where_13 = None
        index_3 = torch.ops.aten.index.Tensor(arg0_1, [view_4, view_5, view_16, view_15]);  arg0_1 = view_4 = view_5 = view_16 = view_15 = None
        mul_14 = torch.ops.aten.mul.Tensor(index_3, view_17);  index_3 = view_17 = None
        add_10 = torch.ops.aten.add.Tensor(add_9, mul_14);  add_9 = mul_14 = None
        return (add_10,)
        
def load_args(reader):
    buf0 = reader.storage(None, 15102720)
    reader.tensor(buf0, (8, 3, 345, 456), is_leaf=True)  # arg0_1
    buf1 = reader.storage(None, 192)
    reader.tensor(buf1, (8, 2, 3), is_leaf=True)  # arg1_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
