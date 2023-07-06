
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



# torch version: 2.1.0a0+git7f9b733
# torch cuda version: 11.7
# torch git version: 7f9b73312a50fbda20483b207b3731fac544504c


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
        upsample_bicubic2d = torch.ops.aten.upsample_bicubic2d.default(arg0_1, [12, 32], False);  arg0_1 = None
        return (upsample_bicubic2d,)
        
def load_args(reader):
    buf0 = reader.storage(None, 12288)
    reader.tensor(buf0, (1, 3, 32, 32), is_leaf=True)  # arg0_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='real')
