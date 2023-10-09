
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



# torch version: 2.2.0a0+git38bb283
# torch cuda version: 12.1
# torch git version: 38bb283a8e6dceb4617bd73263935c7c8fad5fb1


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
        iota = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view = torch.ops.aten.view.default(iota, [2, 1, 1, 1]);  iota = None
        iota_1 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_1 = torch.ops.aten.view.default(iota_1, [1, 3, 1, 1]);  iota_1 = None
        iota_2 = torch.ops.prims.iota.default(272, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        add = torch.ops.aten.add.Tensor(iota_2, 0.5);  iota_2 = None
        mul = torch.ops.aten.mul.Tensor(add, 1.6764705882352942);  add = None
        sub = torch.ops.aten.sub.Tensor(mul, 1.6764705882352942)
        add_1 = torch.ops.aten.add.Tensor(sub, 0.5);  sub = None
        convert_element_type = torch.ops.prims.convert_element_type.default(add_1, torch.int64);  add_1 = None
        clamp_min = torch.ops.aten.clamp_min.default(convert_element_type, 0);  convert_element_type = None
        add_2 = torch.ops.aten.add.Tensor(mul, 1.6764705882352942)
        add_3 = torch.ops.aten.add.Tensor(add_2, 0.5);  add_2 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(add_3, torch.int64);  add_3 = None
        clamp_max = torch.ops.aten.clamp_max.default(convert_element_type_1, 456);  convert_element_type_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(clamp_max, clamp_min);  clamp_max = None
        clamp_max_1 = torch.ops.aten.clamp_max.default(sub_1, 5);  sub_1 = None
        iota_3 = torch.ops.prims.iota.default(5, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_2 = torch.ops.aten.view.default(iota_3, [-1, 1]);  iota_3 = None
        add_4 = torch.ops.aten.add.Tensor(view_2, clamp_min)
        sub_2 = torch.ops.aten.sub.Tensor(add_4, mul);  add_4 = mul = None
        add_5 = torch.ops.aten.add.Tensor(sub_2, 0.5);  sub_2 = None
        mul_1 = torch.ops.aten.mul.Tensor(add_5, 0.5964912280701754);  add_5 = None
        abs_1 = torch.ops.aten.abs.default(mul_1);  mul_1 = None
        clamp_max_2 = torch.ops.aten.clamp_max.default(abs_1, 1.0);  abs_1 = None
        sub_3 = torch.ops.aten.sub.Tensor(1.0, clamp_max_2);  clamp_max_2 = None
        lt = torch.ops.aten.lt.Tensor(view_2, clamp_max_1);  view_2 = clamp_max_1 = None
        full_default = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where = torch.ops.aten.where.self(lt, sub_3, full_default);  lt = sub_3 = full_default = None
        sum_1 = torch.ops.aten.sum.dim_IntList(where, [0])
        div = torch.ops.aten.div.Tensor(where, sum_1);  where = sum_1 = None
        iota_4 = torch.ops.prims.iota.default(345, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_3 = torch.ops.aten.view.default(iota_4, [1, 1, 345, 1]);  iota_4 = None
        view_4 = torch.ops.aten.view.default(clamp_min, [1, 1, 1, 272]);  clamp_min = None
        add_6 = torch.ops.aten.add.Tensor(view_4, 0)
        clamp_max_3 = torch.ops.aten.clamp_max.default(add_6, 455);  add_6 = None
        index = torch.ops.aten.index.Tensor(arg0_1, [view, view_1, view_3, clamp_max_3]);  clamp_max_3 = None
        add_7 = torch.ops.aten.add.Tensor(view_4, 1)
        clamp_max_4 = torch.ops.aten.clamp_max.default(add_7, 455);  add_7 = None
        index_1 = torch.ops.aten.index.Tensor(arg0_1, [view, view_1, view_3, clamp_max_4]);  clamp_max_4 = None
        add_8 = torch.ops.aten.add.Tensor(view_4, 2)
        clamp_max_5 = torch.ops.aten.clamp_max.default(add_8, 455);  add_8 = None
        index_2 = torch.ops.aten.index.Tensor(arg0_1, [view, view_1, view_3, clamp_max_5]);  clamp_max_5 = None
        add_9 = torch.ops.aten.add.Tensor(view_4, 3)
        clamp_max_6 = torch.ops.aten.clamp_max.default(add_9, 455);  add_9 = None
        index_3 = torch.ops.aten.index.Tensor(arg0_1, [view, view_1, view_3, clamp_max_6]);  clamp_max_6 = None
        add_10 = torch.ops.aten.add.Tensor(view_4, 4);  view_4 = None
        clamp_max_7 = torch.ops.aten.clamp_max.default(add_10, 455);  add_10 = None
        index_4 = torch.ops.aten.index.Tensor(arg0_1, [view, view_1, view_3, clamp_max_7]);  arg0_1 = view = view_1 = view_3 = clamp_max_7 = None
        unbind = torch.ops.aten.unbind.int(div);  div = None
        getitem = unbind[0]
        getitem_1 = unbind[1]
        getitem_2 = unbind[2]
        getitem_3 = unbind[3];  unbind = None
        full_default_1 = torch.ops.aten.full.default([272], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        mul_2 = torch.ops.aten.mul.Tensor(index, getitem);  index = getitem = None
        mul_3 = torch.ops.aten.mul.Tensor(index_1, getitem_1);  index_1 = getitem_1 = None
        add_11 = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        mul_4 = torch.ops.aten.mul.Tensor(index_2, getitem_2);  index_2 = getitem_2 = None
        add_12 = torch.ops.aten.add.Tensor(add_11, mul_4);  add_11 = mul_4 = None
        mul_5 = torch.ops.aten.mul.Tensor(index_3, getitem_3);  index_3 = getitem_3 = None
        add_13 = torch.ops.aten.add.Tensor(add_12, mul_5);  add_12 = mul_5 = None
        mul_6 = torch.ops.aten.mul.Tensor(index_4, full_default_1);  index_4 = full_default_1 = None
        add_14 = torch.ops.aten.add.Tensor(add_13, mul_6);  add_13 = mul_6 = None
        iota_5 = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_5 = torch.ops.aten.view.default(iota_5, [2, 1, 1, 1]);  iota_5 = None
        iota_6 = torch.ops.prims.iota.default(3, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_6 = torch.ops.aten.view.default(iota_6, [1, 3, 1, 1]);  iota_6 = None
        iota_7 = torch.ops.prims.iota.default(271, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        add_15 = torch.ops.aten.add.Tensor(iota_7, 0.5);  iota_7 = None
        mul_7 = torch.ops.aten.mul.Tensor(add_15, 1.2730627306273063);  add_15 = None
        sub_4 = torch.ops.aten.sub.Tensor(mul_7, 1.2730627306273063)
        add_16 = torch.ops.aten.add.Tensor(sub_4, 0.5);  sub_4 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(add_16, torch.int64);  add_16 = None
        clamp_min_1 = torch.ops.aten.clamp_min.default(convert_element_type_2, 0);  convert_element_type_2 = None
        add_17 = torch.ops.aten.add.Tensor(mul_7, 1.2730627306273063)
        add_18 = torch.ops.aten.add.Tensor(add_17, 0.5);  add_17 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(add_18, torch.int64);  add_18 = None
        clamp_max_8 = torch.ops.aten.clamp_max.default(convert_element_type_3, 345);  convert_element_type_3 = None
        sub_5 = torch.ops.aten.sub.Tensor(clamp_max_8, clamp_min_1);  clamp_max_8 = None
        clamp_max_9 = torch.ops.aten.clamp_max.default(sub_5, 5);  sub_5 = None
        iota_8 = torch.ops.prims.iota.default(5, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_7 = torch.ops.aten.view.default(iota_8, [-1, 1]);  iota_8 = None
        add_19 = torch.ops.aten.add.Tensor(view_7, clamp_min_1)
        sub_6 = torch.ops.aten.sub.Tensor(add_19, mul_7);  add_19 = mul_7 = None
        add_20 = torch.ops.aten.add.Tensor(sub_6, 0.5);  sub_6 = None
        mul_8 = torch.ops.aten.mul.Tensor(add_20, 0.7855072463768116);  add_20 = None
        abs_2 = torch.ops.aten.abs.default(mul_8);  mul_8 = None
        clamp_max_10 = torch.ops.aten.clamp_max.default(abs_2, 1.0);  abs_2 = None
        sub_7 = torch.ops.aten.sub.Tensor(1.0, clamp_max_10);  clamp_max_10 = None
        lt_1 = torch.ops.aten.lt.Tensor(view_7, clamp_max_9);  view_7 = clamp_max_9 = None
        full_default_2 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where_1 = torch.ops.aten.where.self(lt_1, sub_7, full_default_2);  lt_1 = sub_7 = full_default_2 = None
        sum_2 = torch.ops.aten.sum.dim_IntList(where_1, [0])
        div_1 = torch.ops.aten.div.Tensor(where_1, sum_2);  where_1 = sum_2 = None
        iota_9 = torch.ops.prims.iota.default(272, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_8 = torch.ops.aten.view.default(iota_9, [1, 1, 1, 272]);  iota_9 = None
        view_9 = torch.ops.aten.view.default(clamp_min_1, [1, 1, 271, 1]);  clamp_min_1 = None
        add_21 = torch.ops.aten.add.Tensor(view_9, 0)
        clamp_max_11 = torch.ops.aten.clamp_max.default(add_21, 344);  add_21 = None
        index_5 = torch.ops.aten.index.Tensor(add_14, [view_5, view_6, clamp_max_11, view_8]);  clamp_max_11 = None
        add_22 = torch.ops.aten.add.Tensor(view_9, 1)
        clamp_max_12 = torch.ops.aten.clamp_max.default(add_22, 344);  add_22 = None
        index_6 = torch.ops.aten.index.Tensor(add_14, [view_5, view_6, clamp_max_12, view_8]);  clamp_max_12 = None
        add_23 = torch.ops.aten.add.Tensor(view_9, 2)
        clamp_max_13 = torch.ops.aten.clamp_max.default(add_23, 344);  add_23 = None
        index_7 = torch.ops.aten.index.Tensor(add_14, [view_5, view_6, clamp_max_13, view_8]);  clamp_max_13 = None
        add_24 = torch.ops.aten.add.Tensor(view_9, 3)
        clamp_max_14 = torch.ops.aten.clamp_max.default(add_24, 344);  add_24 = None
        index_8 = torch.ops.aten.index.Tensor(add_14, [view_5, view_6, clamp_max_14, view_8]);  clamp_max_14 = None
        add_25 = torch.ops.aten.add.Tensor(view_9, 4);  view_9 = None
        clamp_max_15 = torch.ops.aten.clamp_max.default(add_25, 344);  add_25 = None
        index_9 = torch.ops.aten.index.Tensor(add_14, [view_5, view_6, clamp_max_15, view_8]);  add_14 = view_5 = view_6 = clamp_max_15 = view_8 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(div_1, -1);  div_1 = None
        unbind_1 = torch.ops.aten.unbind.int(unsqueeze);  unsqueeze = None
        getitem_5 = unbind_1[0]
        getitem_6 = unbind_1[1]
        getitem_7 = unbind_1[2]
        getitem_8 = unbind_1[3]
        getitem_9 = unbind_1[4];  unbind_1 = None
        mul_9 = torch.ops.aten.mul.Tensor(index_5, getitem_5);  index_5 = getitem_5 = None
        mul_10 = torch.ops.aten.mul.Tensor(index_6, getitem_6);  index_6 = getitem_6 = None
        add_26 = torch.ops.aten.add.Tensor(mul_9, mul_10);  mul_9 = mul_10 = None
        mul_11 = torch.ops.aten.mul.Tensor(index_7, getitem_7);  index_7 = getitem_7 = None
        add_27 = torch.ops.aten.add.Tensor(add_26, mul_11);  add_26 = mul_11 = None
        mul_12 = torch.ops.aten.mul.Tensor(index_8, getitem_8);  index_8 = getitem_8 = None
        add_28 = torch.ops.aten.add.Tensor(add_27, mul_12);  add_27 = mul_12 = None
        mul_13 = torch.ops.aten.mul.Tensor(index_9, getitem_9);  index_9 = getitem_9 = None
        add_29 = torch.ops.aten.add.Tensor(add_28, mul_13);  add_28 = mul_13 = None
        clone = torch.ops.aten.clone.default(add_29, memory_format = torch.channels_last);  add_29 = None
        return (clone,)
        
def load_args(reader):
    buf0 = reader.storage(None, 3775680)
    reader.tensor(buf0, (2, 3, 345, 456), (471960, 1, 1368, 3), is_leaf=True)  # arg0_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
