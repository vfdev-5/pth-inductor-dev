# # TORCH_COMPILE_DEBUG=1 python -u check_math_ops.py

import math
import torch


def func(x, a, fn):

    n = int(fn(a) * 2) + 5
    t = torch.ones(n)

    y = x + t
    return y


cfunc = torch.compile(func, dynamic=True, fullgraph=True)

device = "cpu"  # or "cuda"
# device = "cuda"
x = torch.tensor(0, dtype=torch.float32, device=device)
a = 12

for fn in [
    # math.cos,
    # math.cosh,
    math.sqrt,
    math.acos,

    # math.sin,
    # math.sinh,
    # math.asin,

    # math.tan,
    # math.tanh,
    # math.atan,
]:
    b = a
    if fn in (math.acos, math.asin):
        b = -1

    print("-", fn, b)

    out = cfunc(x, b, fn)
    expected = func(x, b, fn)
    torch.testing.assert_close(out, expected)




# expr: ((floor(2*cos(ks0)) + 5)//8) True
#
# expr: ((floor(2*acos(ks0)) + 5)//8) None
#