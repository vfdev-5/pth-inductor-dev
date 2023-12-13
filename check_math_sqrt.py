# # TORCH_COMPILE_DEBUG=1 python -u check_math_sqrt.py

import math
import torch


def func(x, a, fn):

    n = int(fn(a) * 2) + 1
    t = torch.ones(n)

    y = x + t
    return y


cfunc = torch.compile(func, dynamic=True, fullgraph=True)

device = "cpu"  # or "cuda"
# device = "cuda"
x = torch.tensor(0, dtype=torch.float32, device=device)
a = 12

for fn in [
    math.sqrt,
]:
    print("-", fn, a)

    out = cfunc(x, a, fn)
    expected = func(x, a, fn)
    torch.testing.assert_close(out, expected)