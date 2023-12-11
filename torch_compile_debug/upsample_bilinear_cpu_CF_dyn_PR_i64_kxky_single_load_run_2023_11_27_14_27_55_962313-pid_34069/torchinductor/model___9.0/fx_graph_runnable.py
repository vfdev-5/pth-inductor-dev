
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
torch._dynamo.config.assume_static_by_default = False

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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
        sub = arg4_1 - 0
        add = sub + 1;  sub = None
        sub_1 = add - 1;  add = None
        floordiv = sub_1 // 1;  sub_1 = None
        iota = torch.ops.prims.iota.default(floordiv, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        sub_2 = arg5_1 - 0
        add_1 = sub_2 + 1;  sub_2 = None
        sub_3 = add_1 - 1;  add_1 = None
        floordiv_1 = sub_3 // 1;  sub_3 = None
        iota_1 = torch.ops.prims.iota.default(floordiv_1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False);  floordiv_1 = None
        add_2 = torch.ops.aten.add.Tensor(iota_1, 0.5);  iota_1 = None
        sym_size_int = torch.ops.aten.sym_size.int(arg3_1, 3)
        truediv = sym_size_int / arg5_1
        mul = torch.ops.aten.mul.Tensor(add_2, truediv);  add_2 = truediv = None
        sub_4 = torch.ops.aten.sub.Tensor(mul, 0.5);  mul = None
        clamp_min = torch.ops.aten.clamp_min.default(sub_4, 0.0);  sub_4 = None
        add_3 = torch.ops.aten.add.Tensor(iota, 0.5);  iota = None
        truediv_1 = arg2_1 / arg4_1
        mul_1 = torch.ops.aten.mul.Tensor(add_3, truediv_1);  add_3 = truediv_1 = None
        sub_5 = torch.ops.aten.sub.Tensor(mul_1, 0.5);  mul_1 = None
        clamp_min_1 = torch.ops.aten.clamp_min.default(sub_5, 0.0);  sub_5 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(clamp_min, torch.int32)
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(clamp_min_1, torch.int32)
        iota_2 = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        iota_3 = torch.ops.prims.iota.default(2, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        unsqueeze = torch.ops.aten.unsqueeze.default(convert_element_type, -1)
        add_4 = torch.ops.aten.add.Tensor(unsqueeze, iota_2);  unsqueeze = iota_2 = None
        sub_6 = sym_size_int - 1;  sym_size_int = None
        clamp_max = torch.ops.aten.clamp_max.default(add_4, sub_6);  add_4 = sub_6 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(convert_element_type_1, -1)
        add_5 = torch.ops.aten.add.Tensor(unsqueeze_1, iota_3);  unsqueeze_1 = iota_3 = None
        sub_7 = arg2_1 - 1;  arg2_1 = None
        clamp_max_1 = torch.ops.aten.clamp_max.default(add_5, sub_7);  add_5 = sub_7 = None
        sub_8 = torch.ops.aten.sub.Tensor(clamp_min, convert_element_type);  clamp_min = convert_element_type = None
        clamp_min_2 = torch.ops.aten.clamp_min.default(sub_8, 0.0);  sub_8 = None
        clamp_max_2 = torch.ops.aten.clamp_max.default(clamp_min_2, 1.0);  clamp_min_2 = None
        sub_9 = torch.ops.aten.sub.Tensor(1.0, clamp_max_2)
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(sub_9, 1);  sub_9 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(clamp_max_2, 1);  clamp_max_2 = None
        cat = torch.ops.aten.cat.default([unsqueeze_2, unsqueeze_3], -1);  unsqueeze_2 = unsqueeze_3 = None
        sub_10 = torch.ops.aten.sub.Tensor(clamp_min_1, convert_element_type_1);  clamp_min_1 = convert_element_type_1 = None
        clamp_min_3 = torch.ops.aten.clamp_min.default(sub_10, 0.0);  sub_10 = None
        clamp_max_3 = torch.ops.aten.clamp_max.default(clamp_min_3, 1.0);  clamp_min_3 = None
        sub_11 = torch.ops.aten.sub.Tensor(1.0, clamp_max_3)
        unsqueeze_4 = torch.ops.aten.unsqueeze.default(sub_11, 1);  sub_11 = None
        unsqueeze_5 = torch.ops.aten.unsqueeze.default(clamp_max_3, 1);  clamp_max_3 = None
        cat_1 = torch.ops.aten.cat.default([unsqueeze_4, unsqueeze_5], -1);  unsqueeze_4 = unsqueeze_5 = None
        view = torch.ops.aten.view.default(clamp_max_1, [floordiv, 2, 1, 1]);  clamp_max_1 = floordiv = None
        _unsafe_index = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, view, clamp_max]);  arg3_1 = view = clamp_max = None
        view_1 = torch.ops.aten.view.default(cat_1, [arg4_1, 2, 1, 1]);  cat_1 = arg4_1 = None
        view_2 = torch.ops.aten.view.default(cat, [1, arg5_1, 2]);  cat = arg5_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(view_2, _unsafe_index);  view_2 = _unsafe_index = None
        mul_3 = torch.ops.aten.mul.Tensor(view_1, mul_2);  view_1 = mul_2 = None
        sum_1 = torch.ops.aten.sum.dim_IntList(mul_3, [-1, -3]);  mul_3 = None
        round_1 = torch.ops.aten.round.default(sum_1);  sum_1 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(round_1, torch.uint8);  round_1 = None
        return (convert_element_type_2,)
        
def load_args(reader):
    reader.symint(2)  # arg0_1
    reader.symint(3)  # arg1_1
    reader.symint(500)  # arg2_1
    buf0 = reader.storage(None, s0*s1*s2**2, dtype_hint=torch.uint8)
    reader.tensor(buf0, (s0, s1, s2, s2), dtype=torch.uint8, is_leaf=True)  # arg3_1
    reader.symint(256)  # arg4_1
    reader.symint(256)  # arg5_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
