
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
        iota = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type = torch.ops.prims.convert_element_type.default(iota, torch.float32);  iota = None
        mul = torch.ops.aten.mul.Tensor(convert_element_type, 1);  convert_element_type = None
        add = torch.ops.aten.add.Tensor(mul, 0);  mul = None
        mul_1 = torch.ops.aten.mul.Tensor(add, 1.953125);  add = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(mul_1, torch.int64);  mul_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(convert_element_type_1, -1);  convert_element_type_1 = None
        iota_1 = torch.ops.prims.iota.default(256, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(iota_1, torch.float32);  iota_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None
        add_1 = torch.ops.aten.add.Tensor(mul_2, 0);  mul_2 = None
        mul_3 = torch.ops.aten.mul.Tensor(add_1, 1.5625);  add_1 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(mul_3, torch.int64);  mul_3 = None
        _unsafe_index = torch.ops.aten._unsafe_index.Tensor(arg0_1, [None, None, unsqueeze, convert_element_type_3]);  arg0_1 = unsqueeze = convert_element_type_3 = None
        return (_unsafe_index,)
        
def load_args(reader):
    buf0 = reader.storage(None, 2400000, device=device(type='cuda', index=0))
    reader.tensor(buf0, (1, 3, 500, 400), is_leaf=True)  # arg0_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real', check_str=None)
