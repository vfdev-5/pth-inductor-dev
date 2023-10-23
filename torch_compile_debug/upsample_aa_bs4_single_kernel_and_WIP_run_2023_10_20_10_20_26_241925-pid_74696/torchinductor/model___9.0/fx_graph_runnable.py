
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
torch._dynamo.config.cache_size_limit = 100000

torch._functorch.config.debug_partitioner = True


isolate_fails_code_str = None



# torch version: 2.2.0a0+gitb1bf15b
# torch cuda version: 12.1
# torch git version: b1bf15b79357de8e88f97edc2061e6e8a425f40b


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2023 NVIDIA Corporation 
# Built on Mon_Apr__3_17:16:06_PDT_2023 
# Cuda compilation tools, release 12.1, V12.1.105 
# Build cuda_12.1.r12.1/compiler.32688072_0 

# GPU Hardware Info: 
# Quadro RTX 8000 : 2 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1):
        iota = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        add = torch.ops.aten.add.Tensor(iota, 0.5);  iota = None
        mul = torch.ops.aten.mul.Tensor(add, 1.5625);  add = None
        sub = torch.ops.aten.sub.Tensor(mul, 1.5625)
        add_1 = torch.ops.aten.add.Tensor(sub, 0.5);  sub = None
        convert_element_type = torch.ops.prims.convert_element_type.default(add_1, torch.int32);  add_1 = None
        clamp_min = torch.ops.aten.clamp_min.default(convert_element_type, 0);  convert_element_type = None
        add_2 = torch.ops.aten.add.Tensor(mul, 1.5625)
        add_3 = torch.ops.aten.add.Tensor(add_2, 0.5);  add_2 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(add_3, torch.int32);  add_3 = None
        clamp_max = torch.ops.aten.clamp_max.default(convert_element_type_1, 400);  convert_element_type_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(clamp_max, clamp_min);  clamp_max = None
        clamp_max_1 = torch.ops.aten.clamp_max.default(sub_1, 5);  sub_1 = None
        iota_1 = torch.ops.prims.iota.default(5, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        view = torch.ops.aten.view.default(mul, [-1, 1]);  mul = None
        view_1 = torch.ops.aten.view.default(clamp_min, [-1, 1]);  clamp_min = None
        view_2 = torch.ops.aten.view.default(clamp_max_1, [-1, 1]);  clamp_max_1 = None
        add_4 = torch.ops.aten.add.Tensor(iota_1, view_1)
        sub_2 = torch.ops.aten.sub.Tensor(add_4, view);  add_4 = view = None
        add_5 = torch.ops.aten.add.Tensor(sub_2, 0.5);  sub_2 = None
        mul_1 = torch.ops.aten.mul.Tensor(add_5, 0.64);  add_5 = None
        abs_1 = torch.ops.aten.abs.default(mul_1);  mul_1 = None
        clamp_max_2 = torch.ops.aten.clamp_max.default(abs_1, 1.0);  abs_1 = None
        sub_3 = torch.ops.aten.sub.Tensor(1.0, clamp_max_2);  clamp_max_2 = None
        lt = torch.ops.aten.lt.Tensor(iota_1, view_2);  iota_1 = view_2 = None
        full_default = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where = torch.ops.aten.where.self(lt, sub_3, full_default);  lt = sub_3 = full_default = None
        sum_1 = torch.ops.aten.sum.dim_IntList(where, [-1])
        unsqueeze = torch.ops.aten.unsqueeze.default(sum_1, -1);  sum_1 = None
        div = torch.ops.aten.div.Tensor(where, unsqueeze);  where = unsqueeze = None
        view_3 = torch.ops.aten.view.default(view_1, [-1]);  view_1 = None
        iota_2 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        add_6 = torch.ops.aten.add.Tensor(iota_2, 0.5);  iota_2 = None
        mul_2 = torch.ops.aten.mul.Tensor(add_6, 1.953125);  add_6 = None
        sub_4 = torch.ops.aten.sub.Tensor(mul_2, 1.953125)
        add_7 = torch.ops.aten.add.Tensor(sub_4, 0.5);  sub_4 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(add_7, torch.int32);  add_7 = None
        clamp_min_1 = torch.ops.aten.clamp_min.default(convert_element_type_2, 0);  convert_element_type_2 = None
        add_8 = torch.ops.aten.add.Tensor(mul_2, 1.953125)
        add_9 = torch.ops.aten.add.Tensor(add_8, 0.5);  add_8 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(add_9, torch.int32);  add_9 = None
        clamp_max_3 = torch.ops.aten.clamp_max.default(convert_element_type_3, 500);  convert_element_type_3 = None
        sub_5 = torch.ops.aten.sub.Tensor(clamp_max_3, clamp_min_1);  clamp_max_3 = None
        clamp_max_4 = torch.ops.aten.clamp_max.default(sub_5, 5);  sub_5 = None
        iota_3 = torch.ops.prims.iota.default(5, start = 0, step = 1, dtype = torch.int32, device = device(type='cuda', index=0), requires_grad = False)
        view_4 = torch.ops.aten.view.default(mul_2, [-1, 1]);  mul_2 = None
        view_5 = torch.ops.aten.view.default(clamp_min_1, [-1, 1]);  clamp_min_1 = None
        view_6 = torch.ops.aten.view.default(clamp_max_4, [-1, 1]);  clamp_max_4 = None
        add_10 = torch.ops.aten.add.Tensor(iota_3, view_5)
        sub_6 = torch.ops.aten.sub.Tensor(add_10, view_4);  add_10 = view_4 = None
        add_11 = torch.ops.aten.add.Tensor(sub_6, 0.5);  sub_6 = None
        mul_3 = torch.ops.aten.mul.Tensor(add_11, 0.512);  add_11 = None
        abs_2 = torch.ops.aten.abs.default(mul_3);  mul_3 = None
        clamp_max_5 = torch.ops.aten.clamp_max.default(abs_2, 1.0);  abs_2 = None
        sub_7 = torch.ops.aten.sub.Tensor(1.0, clamp_max_5);  clamp_max_5 = None
        lt_1 = torch.ops.aten.lt.Tensor(iota_3, view_6);  iota_3 = view_6 = None
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1 = torch.ops.aten.where.self(lt_1, sub_7, full_default_1);  lt_1 = sub_7 = full_default_1 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(where_1, [-1])
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(sum_2, -1);  sum_2 = None
        div_1 = torch.ops.aten.div.Tensor(where_1, unsqueeze_1);  where_1 = unsqueeze_1 = None
        view_7 = torch.ops.aten.view.default(view_5, [-1]);  view_5 = None
        iota_4 = torch.ops.prims.iota.default(5, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        iota_5 = torch.ops.prims.iota.default(5, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(view_3, -1);  view_3 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(view_7, -1);  view_7 = None
        add_12 = torch.ops.aten.add.Tensor(unsqueeze_2, iota_4);  unsqueeze_2 = iota_4 = None
        clamp_max_6 = torch.ops.aten.clamp_max.default(add_12, 399);  add_12 = None
        add_13 = torch.ops.aten.add.Tensor(unsqueeze_3, iota_5);  unsqueeze_3 = iota_5 = None
        clamp_max_7 = torch.ops.aten.clamp_max.default(add_13, 499);  add_13 = None
        view_8 = torch.ops.aten.view.default(clamp_max_7, [256, 5, 1, 1]);  clamp_max_7 = None
        slice_1 = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, 9223372036854775807);  arg0_1 = None
        slice_2 = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 9223372036854775807);  slice_1 = None
        index = torch.ops.aten.index.Tensor(slice_2, [None, None, view_8, clamp_max_6]);  slice_2 = view_8 = clamp_max_6 = None
        view_9 = torch.ops.aten.view.default(div_1, [256, 5, 1, 1]);  div_1 = None
        view_10 = torch.ops.aten.view.default(div, [1, 256, 5]);  div = None
        mul_4 = torch.ops.aten.mul.Tensor(view_10, index);  view_10 = index = None
        mul_5 = torch.ops.aten.mul.Tensor(view_9, mul_4);  view_9 = mul_4 = None
        sum_3 = torch.ops.aten.sum.dim_IntList(mul_5, [-1, -3]);  mul_5 = None
        return (sum_3,)
        
def load_args(reader):
    buf0 = reader.storage(None, 9600000, device=device(type='cuda', index=0))
    reader.tensor(buf0, (4, 3, 500, 400), is_leaf=True)  # arg0_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
