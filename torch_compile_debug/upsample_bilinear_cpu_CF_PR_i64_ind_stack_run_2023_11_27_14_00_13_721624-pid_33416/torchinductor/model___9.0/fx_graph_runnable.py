
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims

import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config


torch._functorch.config.debug_partitioner = True



isolate_fails_code_str = None



# torch version: 2.2.0a0+git0b5d9e3
# torch cuda version: 12.1
# torch git version: 0b5d9e33c855e1c8d4f4421975abfb7e81fc5026


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

    
    
    def forward(self, arg0_1):
        iota = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        iota_1 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        add = torch.ops.aten.add.Tensor(iota_1, 0.5);  iota_1 = None
        mul = torch.ops.aten.mul.Tensor(add, 1.953125);  add = None
        sub = torch.ops.aten.sub.Tensor(mul, 0.5);  mul = None
        clamp_min = torch.ops.aten.clamp_min.default(sub, 0.0);  sub = None
        add_1 = torch.ops.aten.add.Tensor(iota, 0.5);  iota = None
        mul_1 = torch.ops.aten.mul.Tensor(add_1, 1.953125);  add_1 = None
        sub_1 = torch.ops.aten.sub.Tensor(mul_1, 0.5);  mul_1 = None
        clamp_min_1 = torch.ops.aten.clamp_min.default(sub_1, 0.0);  sub_1 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(clamp_min, torch.int32)
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(clamp_min_1, torch.int32)
        add_2 = torch.ops.aten.add.Tensor(convert_element_type, 1)
        unsqueeze = torch.ops.aten.unsqueeze.default(convert_element_type, 1)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(add_2, 1);  add_2 = None
        cat = torch.ops.aten.cat.default([unsqueeze, unsqueeze_1], -1);  unsqueeze = unsqueeze_1 = None
        clamp_max = torch.ops.aten.clamp_max.default(cat, 499);  cat = None
        add_3 = torch.ops.aten.add.Tensor(convert_element_type_1, 1)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1)
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(add_3, 1);  add_3 = None
        cat_1 = torch.ops.aten.cat.default([unsqueeze_2, unsqueeze_3], -1);  unsqueeze_2 = unsqueeze_3 = None
        clamp_max_1 = torch.ops.aten.clamp_max.default(cat_1, 499);  cat_1 = None
        sub_2 = torch.ops.aten.sub.Tensor(clamp_min, convert_element_type);  clamp_min = convert_element_type = None
        clamp_min_2 = torch.ops.aten.clamp_min.default(sub_2, 0.0);  sub_2 = None
        clamp_max_2 = torch.ops.aten.clamp_max.default(clamp_min_2, 1.0);  clamp_min_2 = None
        sub_3 = torch.ops.aten.sub.Tensor(1.0, clamp_max_2)
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(sub_3, 1);  sub_3 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(clamp_max_2, 1);  clamp_max_2 = None
        cat_2 = torch.ops.aten.cat.default([unsqueeze_4, unsqueeze_5], -1);  unsqueeze_4 = unsqueeze_5 = None
        sub_4 = torch.ops.aten.sub.Tensor(clamp_min_1, convert_element_type_1);  clamp_min_1 = convert_element_type_1 = None
        clamp_min_3 = torch.ops.aten.clamp_min.default(sub_4, 0.0);  sub_4 = None
        clamp_max_3 = torch.ops.aten.clamp_max.default(clamp_min_3, 1.0);  clamp_min_3 = None
        sub_5 = torch.ops.aten.sub.Tensor(1.0, clamp_max_3)
        unsqueeze_6 = torch.ops.aten.unsqueeze.default(sub_5, 1);  sub_5 = None
        unsqueeze_7 = torch.ops.aten.unsqueeze.default(clamp_max_3, 1);  clamp_max_3 = None
        cat_3 = torch.ops.aten.cat.default([unsqueeze_6, unsqueeze_7], -1);  unsqueeze_6 = unsqueeze_7 = None
        view = torch.ops.aten.view.default(clamp_max_1, [256, 2, 1, 1]);  clamp_max_1 = None
        _unsafe_index = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, view, clamp_max]);  arg0_1 = view = clamp_max = None
        view_1 = torch.ops.aten.view.default(cat_3, [256, 2, 1, 1]);  cat_3 = None
        view_2 = torch.ops.aten.view.default(cat_2, [1, 256, 2]);  cat_2 = None
        mul_2 = torch.ops.aten.mul.Tensor(view_2, _unsafe_index);  view_2 = _unsafe_index = None
        mul_3 = torch.ops.aten.mul.Tensor(view_1, mul_2);  view_1 = mul_2 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(mul_3, [-1, -3]);  mul_3 = None
        round_1 = torch.ops.aten.round.default(sum_1);  sum_1 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(round_1, torch.uint8);  round_1 = None
        return (convert_element_type_2,)
        
def load_args(reader):
    buf0 = reader.storage(None, 1500000, dtype_hint=torch.uint8)
    reader.tensor(buf0, (2, 3, 500, 500), dtype=torch.uint8, is_leaf=True)  # arg0_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
