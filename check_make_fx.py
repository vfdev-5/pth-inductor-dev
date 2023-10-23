# import torch
# from torch._decomp import decomposition_table
# from torch._inductor.compile_fx import compile_fx_inner
# from torch.fx.experimental.proxy_tensor import make_fx
# from torch.utils._python_dispatch import TorchDispatchMode


# def fn(theta, size):
#     return torch.nn.functional.affine_grid(theta, size, False)


# t = torch.ones(2, 2, 3)
# size = (2, 3, 32, 32)
# fn_fx = make_fx(fn, decomposition_table=decomposition_table)(*[t, size])

# print("fn_fx:", fn_fx.print_readable())

# class RecordFunctions(TorchDispatchMode):
#     def __torch_dispatch__(self, func, types, args=(), kwargs=None):
#         print("dispatch:", func)
#         return func(*args, **kwargs)

# with RecordFunctions():
#     out = fn_fx(*[t, size])


# fn_compiled = compile_fx_inner(fn_fx, [t, size])
# print("fn_compiled:", fn_compiled)


import torch
from torch._decomp import decomposition_table
from torch.fx.experimental.proxy_tensor import make_fx


def fn(x):
    return x.flip(dims=(-1, ))


x = torch.rand(2, 3, 32, 32)
fn_fx = make_fx(fn, decomposition_table=decomposition_table)(*[x, ])

print("fn_fx:", fn_fx.print_readable())

# class RecordFunctions(TorchDispatchMode):
#     def __torch_dispatch__(self, func, types, args=(), kwargs=None):
#         print("dispatch:", func)
#         return func(*args, **kwargs)

# with RecordFunctions():
#     out = fn_fx(*[x, ])


# fn_compiled = compile_fx_inner(fn_fx, [x, ])
# print("fn_compiled:", fn_compiled)
