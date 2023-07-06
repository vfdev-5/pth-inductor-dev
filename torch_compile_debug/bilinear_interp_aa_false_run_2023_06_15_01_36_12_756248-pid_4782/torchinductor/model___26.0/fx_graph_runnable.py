
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



# torch version: 2.1.0a0+git37359c3
# torch cuda version: 11.7
# torch git version: 37359c36fdb413df3b02996eb0ea2433c147db34


# CUDA Info: 
# nvcc: NVIDIA (R) Cuda compiler driver 
# Copyright (c) 2005-2022 NVIDIA Corporation 
# Built on Wed_Jun__8_16:49:14_PDT_2022 
# Cuda compilation tools, release 11.7, V11.7.99 
# Build cuda_11.7.r11.7/compiler.31442593_0 

# GPU Hardware Info: 
# Quadro RTX 8000 : 2 


from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self):
        super().__init__()

    
    
    def forward(self, arg0_1):
        unsqueeze = torch.ops.aten.unsqueeze.default(arg0_1, 0);  arg0_1 = None
        iota = torch.ops.prims.iota.default(224, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        convert_element_type = torch.ops.prims.convert_element_type.default(iota, torch.float64);  iota = None
        mul = torch.ops.aten.mul.Tensor(convert_element_type, 1);  convert_element_type = None
        add = torch.ops.aten.add.Tensor(mul, 0);  mul = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        iota_1 = torch.ops.prims.iota.default(224, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(iota_1, torch.float64);  iota_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, 0);  mul_1 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        add_2 = torch.ops.aten.add.Tensor(convert_element_type_1, 0.5);  convert_element_type_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(add_2, 1.5401785714285714);  add_2 = None
        sub = torch.ops.aten.sub.Tensor(mul_2, 0.5);  mul_2 = None
        clamp_min = torch.ops.aten.clamp_min.default(sub, 0.0);  sub = None
        add_3 = torch.ops.aten.add.Tensor(convert_element_type_3, 0.5);  convert_element_type_3 = None
        mul_3 = torch.ops.aten.mul.Tensor(add_3, 2.0357142857142856);  add_3 = None
        sub_1 = torch.ops.aten.sub.Tensor(mul_3, 0.5);  mul_3 = None
        clamp_min_1 = torch.ops.aten.clamp_min.default(sub_1, 0.0);  sub_1 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(clamp_min, torch.int64)
        ceil = torch.ops.aten.ceil.default(clamp_min)
        clamp_max = torch.ops.aten.clamp_max.default(ceil, 344);  ceil = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(clamp_max, torch.int64);  clamp_max = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(clamp_min_1, torch.int64)
        ceil_1 = torch.ops.aten.ceil.default(clamp_min_1)
        clamp_max_1 = torch.ops.aten.clamp_max.default(ceil_1, 455);  ceil_1 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.int64);  clamp_max_1 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(clamp_min, 1);  clamp_min = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(convert_element_type_4, 1);  convert_element_type_4 = None
        unsqueeze_3 = torch.ops.aten.unsqueeze.default(convert_element_type_5, 1);  convert_element_type_5 = None
        _unsafe_index = torch.ops.aten._unsafe_index.Tensor(unsqueeze, [None, None, unsqueeze_2, convert_element_type_6])
        _unsafe_index_1 = torch.ops.aten._unsafe_index.Tensor(unsqueeze, [None, None, unsqueeze_3, convert_element_type_6])
        _unsafe_index_2 = torch.ops.aten._unsafe_index.Tensor(unsqueeze, [None, None, unsqueeze_2, convert_element_type_7])
        _unsafe_index_3 = torch.ops.aten._unsafe_index.Tensor(unsqueeze, [None, None, unsqueeze_3, convert_element_type_7]);  unsqueeze = unsqueeze_3 = convert_element_type_7 = None
        sub_2 = torch.ops.aten.sub.Tensor(unsqueeze_1, unsqueeze_2);  unsqueeze_1 = unsqueeze_2 = None
        sub_3 = torch.ops.aten.sub.Tensor(1.0, sub_2)
        sub_4 = torch.ops.aten.sub.Tensor(clamp_min_1, convert_element_type_6);  clamp_min_1 = convert_element_type_6 = None
        sub_5 = torch.ops.aten.sub.Tensor(1.0, sub_4)
        mul_4 = torch.ops.aten.mul.Tensor(_unsafe_index, sub_3);  _unsafe_index = None
        mul_5 = torch.ops.aten.mul.Tensor(_unsafe_index_1, sub_2);  _unsafe_index_1 = None
        add_4 = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        mul_6 = torch.ops.aten.mul.Tensor(_unsafe_index_2, sub_3);  _unsafe_index_2 = sub_3 = None
        mul_7 = torch.ops.aten.mul.Tensor(_unsafe_index_3, sub_2);  _unsafe_index_3 = sub_2 = None
        add_5 = torch.ops.aten.add.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
        mul_8 = torch.ops.aten.mul.Tensor(add_4, sub_5);  add_4 = sub_5 = None
        mul_9 = torch.ops.aten.mul.Tensor(add_5, sub_4);  add_5 = sub_4 = None
        add_6 = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        return (add_6,)
        
def load_args(reader):
    buf0 = reader.storage(None, 1887840)
    reader.tensor(buf0, (3, 345, 456), (1, 1368, 3), is_leaf=True)  # arg0_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real')
