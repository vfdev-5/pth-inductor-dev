
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



# torch version: 2.3.0a0+gitd176294
# torch cuda version: 12.1
# torch git version: d1762941f37a9d1813fba2b14b0d2664fa566fa4


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
        iota = torch.ops.prims.iota.default(floordiv, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False);  floordiv = None
        sub_2 = arg5_1 - 0
        add_1 = sub_2 + 1;  sub_2 = None
        sub_3 = add_1 - 1;  add_1 = None
        floordiv_1 = sub_3 // 1;  sub_3 = None
        iota_1 = torch.ops.prims.iota.default(floordiv_1, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False);  floordiv_1 = None
        sym_size_int = torch.ops.aten.sym_size.int(arg3_1, 3)
        sub_4 = sym_size_int - 1.0
        sub_5 = arg5_1 - 1.0;  arg5_1 = None
        truediv = sub_4 / sub_5;  sub_4 = sub_5 = None
        mul = torch.ops.aten.mul.Tensor(iota_1, truediv);  iota_1 = truediv = None
        clamp_min = torch.ops.aten.clamp_min.default(mul, 0.0);  mul = None
        sym_size_int_1 = torch.ops.aten.sym_size.int(arg3_1, 2)
        sub_6 = sym_size_int_1 - 1.0
        sub_7 = arg4_1 - 1.0;  arg4_1 = None
        truediv_1 = sub_6 / sub_7;  sub_6 = sub_7 = None
        mul_1 = torch.ops.aten.mul.Tensor(iota, truediv_1);  iota = truediv_1 = None
        clamp_min_1 = torch.ops.aten.clamp_min.default(mul_1, 0.0);  mul_1 = None
        unsqueeze = torch.ops.aten.unsqueeze.default(clamp_min_1, -1);  clamp_min_1 = None
        convert_element_type = torch.ops.prims.convert_element_type.default(clamp_min, torch.int64)
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(unsqueeze, torch.int64)
        sub_8 = sym_size_int - 1;  sym_size_int = None
        lt = torch.ops.aten.lt.Scalar(convert_element_type, sub_8);  sub_8 = None
        add_2 = torch.ops.aten.add.Tensor(convert_element_type, 1)
        where = torch.ops.aten.where.self(lt, add_2, convert_element_type);  lt = add_2 = None
        sub_9 = sym_size_int_1 - 1;  sym_size_int_1 = None
        lt_1 = torch.ops.aten.lt.Scalar(convert_element_type_1, sub_9);  sub_9 = None
        add_3 = torch.ops.aten.add.Tensor(convert_element_type_1, 1)
        where_1 = torch.ops.aten.where.self(lt_1, add_3, convert_element_type_1);  lt_1 = add_3 = None
        _unsafe_index = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, convert_element_type_1, convert_element_type])
        _unsafe_index_1 = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, convert_element_type_1, where])
        _unsafe_index_2 = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, where_1, convert_element_type])
        _unsafe_index_3 = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, where_1, where]);  arg3_1 = where_1 = where = None
        sub_10 = torch.ops.aten.sub.Tensor(unsqueeze, convert_element_type_1);  unsqueeze = convert_element_type_1 = None
        clamp_min_2 = torch.ops.aten.clamp_min.default(sub_10, 0.0);  sub_10 = None
        clamp_max = torch.ops.aten.clamp_max.default(clamp_min_2, 1.0);  clamp_min_2 = None
        sub_11 = torch.ops.aten.sub.Tensor(clamp_min, convert_element_type);  clamp_min = convert_element_type = None
        clamp_min_3 = torch.ops.aten.clamp_min.default(sub_11, 0.0);  sub_11 = None
        clamp_max_1 = torch.ops.aten.clamp_max.default(clamp_min_3, 1.0);  clamp_min_3 = None
        sub_12 = torch.ops.aten.sub.Tensor(_unsafe_index_1, _unsafe_index);  _unsafe_index_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_12, clamp_max_1);  sub_12 = None
        add_4 = torch.ops.aten.add.Tensor(_unsafe_index, mul_2);  _unsafe_index = mul_2 = None
        sub_13 = torch.ops.aten.sub.Tensor(_unsafe_index_3, _unsafe_index_2);  _unsafe_index_3 = None
        mul_3 = torch.ops.aten.mul.Tensor(sub_13, clamp_max_1);  sub_13 = clamp_max_1 = None
        add_5 = torch.ops.aten.add.Tensor(_unsafe_index_2, mul_3);  _unsafe_index_2 = mul_3 = None
        sub_14 = torch.ops.aten.sub.Tensor(add_5, add_4);  add_5 = None
        mul_4 = torch.ops.aten.mul.Tensor(sub_14, clamp_max);  sub_14 = clamp_max = None
        add_6 = torch.ops.aten.add.Tensor(add_4, mul_4);  add_4 = mul_4 = None
        return (add_6,)
        
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
