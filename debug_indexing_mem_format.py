# TORCH_COMPILE_DEBUG=1 python debug_indexing_mem_format.py
import torch


from functools import reduce
from typing import Iterable
from torch import Tensor


# def _sum_tensors(ts: Iterable[Tensor]) -> Tensor:
#     return reduce(torch.add, ts)


def func(x, out_size):

    weights = torch.ones(5, out_size, device=x.device)
    src_idx_min = torch.linspace(0, x.shape[-1], steps=out_size, dtype=torch.long, device=x.device)

    max_interp_size = len(weights)
    n, c, in_h, in_w = x.shape
    memory_format = torch.channels_last

    n_idx = torch.arange(n, device=x.device).view(n, 1, 1, 1)
    c_idx = torch.arange(c, device=x.device).view(1, c, 1, 1)
    in_y = torch.arange(in_h, device=x.device).view((1, 1, in_h, 1))

    src_idx_min = src_idx_min.view(1, 1, 1, out_size)
    in_tensor_list = [
        x[
            n_idx, c_idx, in_y, torch.clamp(src_idx_min + k, max=in_w - 1)
        ].contiguous(memory_format=memory_format)
        for k in range(max_interp_size)
    ]

    w_tensor_list = weights.unbind(dim=0)
    output = sum(in_t * w_t for in_t, w_t in zip(in_tensor_list, w_tensor_list))
    output = output.contiguous(memory_format=memory_format)

    return output.sum()


device = "cpu"
memory_format = torch.channels_last

x = torch.rand(1, 3, 345, 456, device=device)
x = x.contiguous(memory_format=memory_format)

out_size = 272
cfunc = torch.compile(func)

out = cfunc(x, out_size)
print(out.shape)
