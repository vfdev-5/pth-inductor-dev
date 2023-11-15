
# import torch


# def func_0(x):
#     if x.is_contiguous(memory_format=torch.channels_last):
#         return x + 1
#     return x + 2


# def func_1(x):
#     numel = x.numel()
#     strides = x.stride()
#     if x.is_contiguous(memory_format=torch.channels_last) and x.shape[0] == 1 and numel != strides[0]:
#         return x + 1
#     return x + 2


# func = func_1

# # cfunc = torch.compile(func, dynamic=True)
# cfunc = torch.compile(func, dynamic=True, fullgraph=True)

# x = torch.rand(1, 3, 32, 32).contiguous(memory_format=torch.channels_last)
# y = cfunc(x)




import torch


# @torch.compile(backend="eager", fullgraph=True)
@torch.compile(backend="eager", dynamic=True, fullgraph=True)
def f(x):
    # numel = x.numel()
    if x.is_contiguous():
    # if numel > 0 and x.is_contiguous():
    # if numel > 0 and x.is_contiguous(memory_format=torch.channels_last):
        return x
    else:
        return 0

x = torch.randn(13, 14)
x = x[::3, ::4]
f(x)

