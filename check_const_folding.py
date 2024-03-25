# Taken from the test test/inductor/test_torchinductor.py
# test_constant_folding_deallocation
# https://github.com/pytorch/pytorch/pull/108421/files

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch._inductor.fx_passes.joint_graph import UniformValueConstantFolder


# def fn(x):
#     x = x * 1.0 + 0.0
#     return x

def fn(x):
    s = x.shape[-1]
    ar = torch.arange(s)
    ar = ar * 1 + 0
    return x + ar


x = torch.rand(100)
mod = make_fx(fn)(x)

nodes_names = [n.name for n in mod.graph.nodes]
print(nodes_names)

cf = UniformValueConstantFolder(mod)
cf.run()

node_replacements = cf.node_replacements
print("node_replacements:", node_replacements)


