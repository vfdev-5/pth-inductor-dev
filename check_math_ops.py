# # TORCH_COMPILE_DEBUG=1 python -u check_math_ops.py
# import math
# import torch




# def func(x, fn, a, b):
#     # c = math.floor(a + 0.5)
#     # c = math.radians(a) + b

#     # c = 0
#     # # for name in ("sqrt", "cos", "cosh", "sin", "sinh", "tan", "tanh", "asin", "acos", "atan", "atan2"):

#     # c = c + fn(a)

#     # c = c + math.sqrt(a)
#     # c = c + math.cos(a)
#     # c = c + math.cosh(a)
#     # c = c + math.sin(a)
#     # c = c + math.sinh(a)
#     # c = c + math.tan(a)
#     # c = c + math.tanh(a)
#     # c = c + math.asin(b)
#     # c = c + math.acos(b)
#     # c = c + math.atan(a)

#     # y = x + c
#     # return y

#     return x + math.sqrt(a)

# # device = "cuda"
# device = "cpu"

# cfunc = torch.compile(func, dynamic=True, fullgraph=True)

# x = torch.tensor([0, 1, 2, 3], dtype=torch.float32, device=device)
# a = 4
# b = 1

# f = math.cos

# out = cfunc(x, f, a, b)
# print(out)
# expected = func(x, f, a, b)

# torch.testing.assert_close(out, expected)


import math
import torch

def func(x, a, b):
    c = 0
    c = c + math.sqrt(a)
    c = c + math.cos(a)
    c = c + math.cosh(a)
    c = c + math.sin(a)
    c = c + math.sinh(a)
    c = c + math.tan(a)
    c = c + math.tanh(a)
    c = c + math.asin(b)
    c = c + math.acos(b)
    c = c + math.atan(a)
    y = x + c
    return y


cfunc = torch.compile(func, dynamic=True, fullgraph=True)

# device = "cpu"  # or "cuda"
device = "cuda"
x = torch.tensor([0, 1, 2, 3], dtype=torch.float32, device=device)
a = 12
b = -1

out = cfunc(x, a, b)
expected = func(x, a, b)
torch.testing.assert_close(out, expected)