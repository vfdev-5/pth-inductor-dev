
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
        iota = torch.ops.prims.iota.default(1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        full_default = torch.ops.aten.full.default([1, 1, 1, 1], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
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
        full_default_1 = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        where = torch.ops.aten.where.self(lt, sub_3, full_default_1);  lt = sub_3 = full_default_1 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(where, [0])
        div = torch.ops.aten.div.Tensor(where, sum_1);  where = sum_1 = None
        iota_4 = torch.ops.prims.iota.default(345, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        view_3 = torch.ops.aten.view.default(iota_4, [1, 1, 345, 1]);  iota_4 = None
        view_4 = torch.ops.aten.view.default(clamp_min, [1, 1, 1, 272]);  clamp_min = None



        add_6 = torch.ops.aten.add.Tensor(view_4, 0)
        clamp_max_3 = torch.ops.aten.clamp_max.default(add_6, 455);  add_6 = None

        index = torch.ops.aten.index.Tensor(arg0_1, [full_default, view_1, view_3, clamp_max_3]);  clamp_max_3 = None
        clone = torch.ops.aten.clone.default(index, memory_format = torch.channels_last);  index = None

        add_7 = torch.ops.aten.add.Tensor(view_4, 1)
        clamp_max_4 = torch.ops.aten.clamp_max.default(add_7, 455);  add_7 = None

        index_1 = torch.ops.aten.index.Tensor(arg0_1, [full_default, view_1, view_3, clamp_max_4]);  clamp_max_4 = None
        clone_1 = torch.ops.aten.clone.default(index_1, memory_format = torch.channels_last);  index_1 = None

        add_8 = torch.ops.aten.add.Tensor(view_4, 2)
        clamp_max_5 = torch.ops.aten.clamp_max.default(add_8, 455);  add_8 = None

        index_2 = torch.ops.aten.index.Tensor(arg0_1, [full_default, view_1, view_3, clamp_max_5]);  clamp_max_5 = None
        clone_2 = torch.ops.aten.clone.default(index_2, memory_format = torch.channels_last);  index_2 = None

        add_9 = torch.ops.aten.add.Tensor(view_4, 3)
        clamp_max_6 = torch.ops.aten.clamp_max.default(add_9, 455);  add_9 = None

        index_3 = torch.ops.aten.index.Tensor(arg0_1, [full_default, view_1, view_3, clamp_max_6]);  clamp_max_6 = None
        clone_3 = torch.ops.aten.clone.default(index_3, memory_format = torch.channels_last);  index_3 = None

        add_10 = torch.ops.aten.add.Tensor(view_4, 4);  view_4 = None
        clamp_max_7 = torch.ops.aten.clamp_max.default(add_10, 455);  add_10 = None

        index_4 = torch.ops.aten.index.Tensor(arg0_1, [full_default, view_1, view_3, clamp_max_7]);  arg0_1 = full_default = view_1 = view_3 = clamp_max_7 = None
        clone_4 = torch.ops.aten.clone.default(index_4, memory_format = torch.channels_last);  index_4 = None



        unbind = torch.ops.aten.unbind.int(div);  div = None
        getitem = unbind[0]
        getitem_1 = unbind[1]
        getitem_2 = unbind[2]
        getitem_3 = unbind[3];  unbind = None


        full_default_2 = torch.ops.aten.full.default([272], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cpu'), pin_memory = False)
        mul_2 = torch.ops.aten.mul.Tensor(clone, getitem);  clone = getitem = None
        mul_3 = torch.ops.aten.mul.Tensor(clone_1, getitem_1);  clone_1 = getitem_1 = None
        add_11 = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        mul_4 = torch.ops.aten.mul.Tensor(clone_2, getitem_2);  clone_2 = getitem_2 = None
        add_12 = torch.ops.aten.add.Tensor(add_11, mul_4);  add_11 = mul_4 = None
        mul_5 = torch.ops.aten.mul.Tensor(clone_3, getitem_3);  clone_3 = getitem_3 = None
        add_13 = torch.ops.aten.add.Tensor(add_12, mul_5);  add_12 = mul_5 = None
        mul_6 = torch.ops.aten.mul.Tensor(clone_4, full_default_2);  clone_4 = full_default_2 = None
        add_14 = torch.ops.aten.add.Tensor(add_13, mul_6);  add_13 = mul_6 = None
        return (add_14,)

def load_args(reader):
    buf0 = reader.storage(None, 1887840)
    reader.tensor(buf0, (1, 3, 345, 456), (471960, 1, 1368, 3), is_leaf=True)  # arg0_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
