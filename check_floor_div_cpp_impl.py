# # TORCH_COMPILE_DEBUG=1 python -u check_floor_div_cpp_impl.py

import torch

aten = torch.ops.aten

# def func(x, a):
#     n = (a * 1.234) // 8.234
#     y = x + n
#     return y

def func(x, a):
    return aten.div(x, a, rounding_mode=None)


cfunc = torch.compile(func, dynamic=True, fullgraph=True)

device = "cpu"  # or "cuda"
# device = "cuda"
# x = torch.tensor(0, dtype=torch.float32, device=device)
x = 124
# x = torch.randint(2**32, 2**40, [100, 100])
a = 33
# a = torch.tensor(33.0, dtype=torch.float64)
a = torch.randint(-10, -1, [100, 100])


out = cfunc(x, a)
expected = func(x, a)

# print(out, expected)

torch.testing.assert_close(out, expected)