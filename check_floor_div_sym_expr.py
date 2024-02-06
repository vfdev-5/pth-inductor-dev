# # TORCH_COMPILE_DEBUG=1 python -u check_floor_div_sym_expr.py

import torch

aten = torch.ops.aten

def func(x, a):
    return aten.div(x * 0.5, a * 1.0, rounding_mode=None)


cfunc = torch.compile(func, dynamic=True, fullgraph=True)

device = "cpu"  # or "cuda"
x = 124
a = 33
# a = torch.randint(-10, -1, [100, 100])

out = cfunc(x, a)
expected = func(x, a)

torch.testing.assert_close(out, expected)