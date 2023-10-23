
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
        iota = torch.ops.prims.iota.default(272, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add = torch.ops.aten.add.Tensor(iota, 0.5);  iota = None
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
        iota_1 = torch.ops.prims.iota.default(5, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view = torch.ops.aten.view.default(mul, [-1, 1]);  mul = None
        view_1 = torch.ops.aten.view.default(clamp_min, [-1, 1]);  clamp_min = None
        view_2 = torch.ops.aten.view.default(clamp_max_1, [-1, 1]);  clamp_max_1 = None
        add_4 = torch.ops.aten.add.Tensor(iota_1, view_1)
        sub_2 = torch.ops.aten.sub.Tensor(add_4, view);  add_4 = view = None
        add_5 = torch.ops.aten.add.Tensor(sub_2, 0.5);  sub_2 = None
        mul_1 = torch.ops.aten.mul.Tensor(add_5, 0.5964912280701754);  add_5 = None
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
        iota_2 = torch.ops.prims.iota.default(5, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(view_3, -1);  view_3 = None
        add_6 = torch.ops.aten.add.Tensor(unsqueeze_1, iota_2);  unsqueeze_1 = iota_2 = None
        clamp_max_3 = torch.ops.aten.clamp_max.default(add_6, 455);  add_6 = None
        slice_1 = torch.ops.aten.slice.Tensor(arg0_1, 0, 0, 9223372036854775807);  arg0_1 = None
        slice_2 = torch.ops.aten.slice.Tensor(slice_1, 1, 0, 9223372036854775807);  slice_1 = None
        slice_3 = torch.ops.aten.slice.Tensor(slice_2, 2, 0, 9223372036854775807);  slice_2 = None
        index = torch.ops.aten.index.Tensor(slice_3, [None, None, None, clamp_max_3]);  slice_3 = clamp_max_3 = None
        mul_2 = torch.ops.aten.mul.Tensor(div, index);  div = index = None
        sum_2 = torch.ops.aten.sum.dim_IntList(mul_2, [-1]);  mul_2 = None
        return (sum_2,)
        
def load_args(reader):
    buf0 = reader.storage(None, 7551360, device=device(type='cuda', index=0))
    reader.tensor(buf0, (4, 3, 345, 456), is_leaf=True)  # arg0_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
