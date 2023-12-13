
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



# torch version: 2.3.0a0+gitde89a53
# torch cuda version: 12.1
# torch git version: de89a53df8222148460e8d6fc49009c0e1c90900


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
        iota = torch.ops.prims.iota.default(ceil, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False);  ceil = None
        convert_element_type = torch.ops.prims.convert_element_type.default(iota, torch.float64);  iota = None
        mul = torch.ops.aten.mul.Tensor(convert_element_type, 1);  convert_element_type = None
        add = torch.ops.aten.add.Tensor(mul, 0);  mul = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(add, torch.float32);  add = None
        sub_1 = arg5_1 - 0
        truediv_1 = sub_1 / 1;  sub_1 = None
        ceil_1 = math_ceil(truediv_1);  truediv_1 = None
        iota_1 = torch.ops.prims.iota.default(ceil_1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False);  ceil_1 = None
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(iota_1, torch.float64);  iota_1 = None
        mul_1 = torch.ops.aten.mul.Tensor(convert_element_type_2, 1);  convert_element_type_2 = None
        add_1 = torch.ops.aten.add.Tensor(mul_1, 0);  mul_1 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(add_1, torch.float32);  add_1 = None
        sym_size_int = torch.ops.aten.sym_size.int(arg3_1, 2)
        sub_2 = sym_size_int - 1
        sub_3 = arg4_1 - 1;  arg4_1 = None
        truediv_2 = sub_2 / sub_3;  sub_2 = sub_3 = None
        mul_2 = torch.ops.aten.mul.Tensor(convert_element_type_1, truediv_2);  convert_element_type_1 = truediv_2 = None
        sym_size_int_1 = torch.ops.aten.sym_size.int(arg3_1, 3)
        sub_4 = sym_size_int_1 - 1
        sub_5 = arg5_1 - 1;  arg5_1 = None
        truediv_3 = sub_4 / sub_5;  sub_4 = sub_5 = None
        mul_3 = torch.ops.aten.mul.Tensor(convert_element_type_3, truediv_3);  convert_element_type_3 = truediv_3 = None
        convert_element_type_4 = torch.ops.prims.convert_element_type.default(mul_2, torch.int64)
        ceil_2 = torch.ops.aten.ceil.default(mul_2)
        sub_6 = sym_size_int - 1;  sym_size_int = None
        clamp_max = torch.ops.aten.clamp_max.default(ceil_2, sub_6);  ceil_2 = sub_6 = None
        convert_element_type_5 = torch.ops.prims.convert_element_type.default(clamp_max, torch.int64);  clamp_max = None
        convert_element_type_6 = torch.ops.prims.convert_element_type.default(mul_3, torch.int64)
        ceil_3 = torch.ops.aten.ceil.default(mul_3)
        sub_7 = sym_size_int_1 - 1;  sym_size_int_1 = None
        clamp_max_1 = torch.ops.aten.clamp_max.default(ceil_3, sub_7);  ceil_3 = sub_7 = None
        convert_element_type_7 = torch.ops.prims.convert_element_type.default(clamp_max_1, torch.int64);  clamp_max_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(mul_2, 1);  mul_2 = None
        unsqueeze_1 = torch.ops.aten.unsqueeze.default(convert_element_type_4, 1);  convert_element_type_4 = None
        unsqueeze_2 = torch.ops.aten.unsqueeze.default(convert_element_type_5, 1);  convert_element_type_5 = None
        _unsafe_index = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, unsqueeze_1, convert_element_type_6])
        _unsafe_index_1 = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, unsqueeze_2, convert_element_type_6])
        _unsafe_index_2 = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, unsqueeze_1, convert_element_type_7])
        _unsafe_index_3 = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, unsqueeze_2, convert_element_type_7]);  arg3_1 = unsqueeze_2 = convert_element_type_7 = None
        sub_8 = torch.ops.aten.sub.Tensor(unsqueeze, unsqueeze_1);  unsqueeze = unsqueeze_1 = None
        sub_9 = torch.ops.aten.sub.Tensor(1.0, sub_8)
        sub_10 = torch.ops.aten.sub.Tensor(mul_3, convert_element_type_6);  mul_3 = convert_element_type_6 = None
        sub_11 = torch.ops.aten.sub.Tensor(1.0, sub_10)
        mul_4 = torch.ops.aten.mul.Tensor(_unsafe_index, sub_9);  _unsafe_index = None
        mul_5 = torch.ops.aten.mul.Tensor(_unsafe_index_1, sub_8);  _unsafe_index_1 = None
        add_2 = torch.ops.aten.add.Tensor(mul_4, mul_5);  mul_4 = mul_5 = None
        mul_6 = torch.ops.aten.mul.Tensor(_unsafe_index_2, sub_9);  _unsafe_index_2 = sub_9 = None
        mul_7 = torch.ops.aten.mul.Tensor(_unsafe_index_3, sub_8);  _unsafe_index_3 = sub_8 = None
        add_3 = torch.ops.aten.add.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
        mul_8 = torch.ops.aten.mul.Tensor(add_2, sub_11);  add_2 = sub_11 = None
        mul_9 = torch.ops.aten.mul.Tensor(add_3, sub_10);  add_3 = sub_10 = None
        add_4 = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        return (add_4,)
        
def load_args(reader):
    reader.symint(3)  # arg0_1
    reader.symint(500)  # arg1_1
    reader.symint(400)  # arg2_1
    buf0 = reader.storage(None, 4*s0*s1*s2)
    reader.tensor(buf0, (1, s0, s1, s2), is_leaf=True)  # arg3_1
    reader.symint(300)  # arg4_1
    reader.symint(256)  # arg5_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
