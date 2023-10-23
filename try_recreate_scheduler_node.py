# TORCH_COMPILE_DEBUG=1 python -u try_recreate_scheduler_node.py

import torch
from torch._inductor.utils import add_scheduler_init_hook


def f(t):
    indices = [
        [1, 3, 5, 7, 9],
        [2, 4, 6, 8, 10],
        [3, 5, 7, 9, 11],
        [4, 6, 8, 10, 12],
    ]
    weights = 1.0 / 5.0 * torch.ones(len(indices), 5, 1)
    src = t[:, :, indices, :]
    out = (src * weights).sum(-2)
    if t.is_contiguous(memory_format=torch.channels_last):
        out = out.contiguous(memory_format=torch.channels_last)
    return out


x = torch.rand(2, 3, 32, 32)
x = x.to(memory_format=torch.channels_last, copy=True)

out = f(x)
print("1", out.shape, out.stride())

def pre_hook_fn(scheduler, nodes):
    print("PRE: --", type(scheduler), len(nodes))
    for node in nodes:
        print("pre -", type(node), node)

def post_hook_fn(scheduler, nodes):
    print("POST: --", type(scheduler), len(scheduler.nodes))
    for node in scheduler.nodes:
        print("\npost -", type(node), node.debug_str())

with add_scheduler_init_hook(pre_hook_fn, post_hook_fn):
    out = torch.compile(f, fullgraph=True)(x)

print("2", out.shape, out.stride())