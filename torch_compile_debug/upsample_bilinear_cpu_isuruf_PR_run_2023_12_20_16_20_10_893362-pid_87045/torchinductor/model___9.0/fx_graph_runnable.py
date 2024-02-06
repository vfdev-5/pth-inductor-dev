
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



# torch version: 2.2.0a0+git3c83df9
# torch cuda version: 12.1
# torch git version: 3c83df9a64c40469346e094d935a567bedd5a457


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

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
        iota = torch.ops.prims.iota.default(1234, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        convert_element_type = torch.ops.prims.convert_element_type.default(iota, torch.float32);  iota = None
        add = torch.ops.aten.add.Tensor(convert_element_type, 0.5);  convert_element_type = None
        sym_size_int = torch.ops.aten.sym_size.int(arg3_1, 2)
        truediv = sym_size_int / 1234
        mul = torch.ops.aten.mul.Tensor(add, truediv);  add = truediv = None
        sub = torch.ops.aten.sub.Tensor(mul, 0.5);  mul = None
        clamp_min = torch.ops.aten.clamp_min.default(sub, 0.0);  sub = None
        view = torch.ops.aten.view.default(clamp_min, [1234, 1]);  clamp_min = None
        convert_element_type_1 = torch.ops.prims.convert_element_type.default(view, torch.int64)
        sub_1 = sym_size_int - 1;  sym_size_int = None
        lt = torch.ops.aten.lt.Scalar(convert_element_type_1, sub_1);  sub_1 = None
        add_1 = torch.ops.aten.add.Tensor(convert_element_type_1, 1)
        where = torch.ops.aten.where.self(lt, add_1, convert_element_type_1);  lt = add_1 = None
        iota_1 = torch.ops.prims.iota.default(1345, start = 0, step = 1, dtype = torch.int64, device = device(type='cpu'), requires_grad = False)
        convert_element_type_2 = torch.ops.prims.convert_element_type.default(iota_1, torch.float32);  iota_1 = None
        add_2 = torch.ops.aten.add.Tensor(convert_element_type_2, 0.5);  convert_element_type_2 = None
        sym_size_int_1 = torch.ops.aten.sym_size.int(arg3_1, 3)
        truediv_1 = sym_size_int_1 / 1345
        mul_1 = torch.ops.aten.mul.Tensor(add_2, truediv_1);  add_2 = truediv_1 = None
        sub_2 = torch.ops.aten.sub.Tensor(mul_1, 0.5);  mul_1 = None
        clamp_min_1 = torch.ops.aten.clamp_min.default(sub_2, 0.0);  sub_2 = None
        convert_element_type_3 = torch.ops.prims.convert_element_type.default(clamp_min_1, torch.int64)
        sub_3 = sym_size_int_1 - 1;  sym_size_int_1 = None
        lt_1 = torch.ops.aten.lt.Scalar(convert_element_type_3, sub_3);  sub_3 = None
        add_3 = torch.ops.aten.add.Tensor(convert_element_type_3, 1)
        where_1 = torch.ops.aten.where.self(lt_1, add_3, convert_element_type_3);  lt_1 = add_3 = None
        _unsafe_index = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, convert_element_type_1, convert_element_type_3])
        _unsafe_index_1 = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, convert_element_type_1, where_1])
        _unsafe_index_2 = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, where, convert_element_type_3])
        _unsafe_index_3 = torch.ops.aten._unsafe_index.Tensor(arg3_1, [None, None, where, where_1]);  arg3_1 = where = where_1 = None
        sub_4 = torch.ops.aten.sub.Tensor(clamp_min_1, convert_element_type_3);  clamp_min_1 = convert_element_type_3 = None
        clamp_min_2 = torch.ops.aten.clamp_min.default(sub_4, 0.0);  sub_4 = None
        clamp_max = torch.ops.aten.clamp_max.default(clamp_min_2, 1.0);  clamp_min_2 = None
        sub_5 = torch.ops.aten.sub.Tensor(_unsafe_index_1, _unsafe_index);  _unsafe_index_1 = None
        mul_2 = torch.ops.aten.mul.Tensor(sub_5, clamp_max);  sub_5 = None
        add_4 = torch.ops.aten.add.Tensor(_unsafe_index, mul_2);  _unsafe_index = mul_2 = None
        sub_6 = torch.ops.aten.sub.Tensor(_unsafe_index_3, _unsafe_index_2);  _unsafe_index_3 = None
        mul_3 = torch.ops.aten.mul.Tensor(sub_6, clamp_max);  sub_6 = clamp_max = None
        add_5 = torch.ops.aten.add.Tensor(_unsafe_index_2, mul_3);  _unsafe_index_2 = mul_3 = None
        sub_7 = torch.ops.aten.sub.Tensor(view, convert_element_type_1);  view = convert_element_type_1 = None
        clamp_min_3 = torch.ops.aten.clamp_min.default(sub_7, 0.0);  sub_7 = None
        clamp_max_1 = torch.ops.aten.clamp_max.default(clamp_min_3, 1.0);  clamp_min_3 = None
        sub_8 = torch.ops.aten.sub.Tensor(add_5, add_4);  add_5 = None
        mul_4 = torch.ops.aten.mul.Tensor(sub_8, clamp_max_1);  sub_8 = clamp_max_1 = None
        add_6 = torch.ops.aten.add.Tensor(add_4, mul_4);  add_4 = mul_4 = None
        return (add_6,)
        
def load_args(reader):
    reader.symint(3)  # arg0_1
    reader.symint(2345)  # arg1_1
    reader.symint(2456)  # arg2_1
    buf0 = reader.storage(None, 4*s0*s1*s2)
    reader.tensor(buf0, (1, s0, s1, s2), is_leaf=True)  # arg3_1
    reader.symint(1234)  # arg4_1
    reader.symint(1345)  # arg5_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
