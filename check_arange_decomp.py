# TORCH_COMPILE_DEBUG=1 python -u check_arange_decomp.py

import torch


# def func(x):
#     s = x.shape[-1]
#     a = torch.arange(s, dtype=torch.float32)
#     b = torch.div(240, s)
#     return s + a + b


# def func(x):
#     s = x.shape[-1]
#     a = torch.arange(s, dtype=torch.float32)
#     return s + a


def func(x):
    s = x.shape[-1]
    a = torch.arange(s).to(dtype=torch.float32)
    return s + a


c_func = torch.compile(func, backend="inductor")

out = c_func(torch.rand(10))

print(out.shape)

