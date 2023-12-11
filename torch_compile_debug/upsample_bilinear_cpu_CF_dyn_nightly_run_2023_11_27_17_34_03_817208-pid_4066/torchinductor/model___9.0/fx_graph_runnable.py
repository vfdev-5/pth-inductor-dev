
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



# torch version: 2.2.0a0+git9fcf1f9
# torch cuda version: 12.1
# torch git version: 9fcf1f9632fc1981e87a5948e5555d05896217b7


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
        truediv = sub / 1;  sub = None
        ceil = math_ceil(truediv);  truediv = None
        iota = torch.ops.prims.iota.default(ceil, start = 0, step = 1, dtype = torch.uint8, device = device(type='cpu'), requires_grad = False);  ceil = None
        sub_1 = arg5_1 - 0
        truediv_1 = sub_1 / 1;  sub_1 = None
        ceil_1 = math_ceil(truediv_1);  truediv_1 = None
        iota_1 = torch.ops.prims.iota.default(ceil_1, start = 0, step = 1, dtype = torch.uint8, device = device(type='cpu'), requires_grad = False);  ceil_1 = None
        add = torch.ops.aten.add.Tensor(iota, 0.5);  iota = None
        truediv_2 = arg1_1 / arg4_1;  arg4_1 = None
        mul = torch.ops.aten.mul.Tensor(add, truediv_2);  add = truediv_2 = None
        sub_2 = torch.ops.aten.sub.Tensor(mul, 0.5);  mul = None
        clamp_min = torch.ops.aten.clamp_min.default(sub_2, 0.0);  sub_2 = None
        add_1 = torch.ops.aten.add.Tensor(iota_1, 0.5);  iota_1 = None
        truediv_3 = arg2_1 / arg5_1;  arg5_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(add_1, truediv_3);  add_1 = truediv_3 = None
        sub_3 = torch.ops.aten.sub.Tensor(mul_1, 0.5);  mul_1 = None
        clamp_min_1 = torch.ops.aten.clamp_min.default(sub_3, 0.0);  sub_3 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(clamp_min, torch.int64)
        ceil_2 = torch.ops.aten.ceil.default(clamp_min)
        sub_4 = arg1_1 - 1;  arg1_1 = None
        clamp_max = torch.ops.aten.clamp_max.default(ceil_2, sub_4);  ceil_2 = sub_4 = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(clamp_max, torch.int64);  clamp_max = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(clamp_min_1, torch.int64)
        ceil_3 = torch.ops.aten.ceil.default(clamp_min_1)
        sub_5 = arg2_1 - 1;  arg2_1 = None
        clamp_max_1 = torch.ops.aten.clamp_max.default(ceil_3, sub_5);  ceil_3 = sub_5 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.int64);  clamp_max_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(clamp_min, 1);  clamp_min = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(convert_element_type, 1);  convert_element_type = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(convert_element_type_1, 1);  convert_element_type_1 = None
        _unsafe_index = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, unsqueeze_1, convert_element_type_2])
        _unsafe_index_1 = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, unsqueeze_2, convert_element_type_2])
        _unsafe_index_2 = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, unsqueeze_1, convert_element_type_3])
        _unsafe_index_3 = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, unsqueeze_2, convert_element_type_3]);  arg3_1 = unsqueeze_2 = convert_element_type_3 = None
        sub_6 = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        sub_7 = torch.ops.aten.sub.Tensor(1.0, sub_6)
        sub_8 = torch.ops.aten.sub.Tensor(clamp_min_1, convert_element_type_2);  clamp_min_1 = convert_element_type_2 = None
        sub_9 = torch.ops.aten.sub.Tensor(1.0, sub_8)
        mul_2 = torch.ops.aten.mul.Tensor(_unsafe_index, sub_7);  _unsafe_index = None
        mul_3 = torch.ops.aten.mul.Tensor(_unsafe_index_1, sub_6);  _unsafe_index_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul_2, mul_3);  mul_2 = mul_3 = None
        mul_4 = torch.ops.aten.mul.Tensor(_unsafe_index_2, sub_7);  _unsafe_index_2 = sub_7 = None
        mul_5 = torch.ops.aten.mul.Tensor(_unsafe_index_3, sub_6);  _unsafe_index_3 = sub_6 = None
        add_3 = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        mul_6 = torch.ops.aten.mul.Tensor(add_2, sub_9);  add_2 = sub_9 = None
        mul_7 = torch.ops.aten.mul.Tensor(add_3, sub_8);  add_3 = sub_8 = None
        add_4 = torch.ops.aten.add.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(add_4, torch.uint8);  add_4 = None
        return (convert_element_type_4,)
        
def load_args(reader):
    reader.symint(3)  # arg0_1
    reader.symint(500)  # arg1_1
    reader.symint(400)  # arg2_1
    buf0 = reader.storage(None, s0*s1*s2, dtype_hint=torch.uint8)
    reader.tensor(buf0, (1, s0, s1, s2), dtype=torch.uint8, is_leaf=True)  # arg3_1
    reader.symint(256)  # arg4_1
    reader.symint(256)  # arg5_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
