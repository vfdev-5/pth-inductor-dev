
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




# import torch

# # @torch.compile(backend="eager", fullgraph=True)
# @torch.compile(backend="eager", dynamic=True, fullgraph=True)
# def f(x):
#     # numel = x.numel()
#     if x.is_contiguous():
#     # if numel > 0 and x.is_contiguous():
#     # if numel > 0 and x.is_contiguous(memory_format=torch.channels_last):
#         return x
#     else:
#         return 0

# x = torch.randn(13, 14)
# x = x[::3, ::4]
# f(x)


# import torch

# def func(x):
#     if x.is_contiguous():
#         return x + 1
#     elif x.is_contiguous(memory_format=torch.channels_last):
#         return x + 2
#     else:
#         return 0

# x = torch.rand(1, 3, 32, 32)

# graph, guards = torch._dynamo.export(func)(x)
# print("\n--- graph:")
# graph.print_readable()

# print("\n--- guards:", type(guards))

# guard_code = []
# for guard in guards:
#     if guard.code_list:
#         guard_code += guard.code_list

# print("\n".join(guard_code))



######## correctness test

# import torch

# def func(x):
#     if x.is_contiguous():
#         return x + 1
#     elif x.is_contiguous(memory_format=torch.channels_last):
#         return x + 2
#     else:
#         return 0

# expected = []
# data = [
#     torch.rand(100),
#     torch.rand(2, 3, 500, 400),
#     torch.rand(2, 3, 500, 400).contiguous(memory_format=torch.channels_last),
#     torch.rand(100)[::2],
#     torch.rand(2, 3, 500, 400)[:, :, 10:-10, 12:-12],
#     torch.rand(2, 3, 500, 400).contiguous(memory_format=torch.channels_last)[:, :, 10:-10, 12:-12],
# ]

# for x in data:
#     expected.append(func(x))

# torch._dynamo.reset()
# cfunc_static_shapes = torch.compile(func, backend="eager", dynamic=False, fullgraph=True)
# output_static = [cfunc_static_shapes(x) for x in data]

# torch._dynamo.reset()
# cfunc_dynamic_shapes = torch.compile(func, backend="eager", dynamic=True, fullgraph=True)
# output_dynamic = [cfunc_dynamic_shapes(x) for x in data]

# for i, (e, o1, o2) in enumerate(zip(expected, output_static, output_dynamic)):
#     print("- i:", i)
#     torch.testing.assert_close(e, o1)
#     torch.testing.assert_close(e, o2)


######## compile counts check

import torch
import torch._dynamo.testing

# We need to set cache size very large to avoid running eager mode as compiled
torch._dynamo.config.cache_size_limit = 100000


def func(x):
    if x.is_contiguous():
        return x + 1
    elif x.is_contiguous(memory_format=torch.channels_last):
        return x + 2
    else:
        return x + 3


data = [
    torch.rand(100),
    torch.rand(2, 3, 500, 400),
    torch.rand(2, 3, 500, 400).contiguous(memory_format=torch.channels_last),
    torch.rand(100)[::2],
    torch.rand(50),
    torch.rand(2, 3, 400, 300),
    torch.rand(2, 3, 400, 300).contiguous(memory_format=torch.channels_last),
    torch.rand(50)[::2],
    torch.rand(2, 3, 500, 400)[:, :, 10:-10, 12:-12],
    torch.rand(2, 3, 500, 400).contiguous(memory_format=torch.channels_last)[:, :, 10:-10, 12:-12],
]

torch._dynamo.reset()
cnt = torch._dynamo.testing.CompileCounter()
cfunc = torch.compile(func, backend=cnt, dynamic=False, fullgraph=True)

assert cnt.frame_count == 0
expected_frame_counts = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10
]

print("--- Check static shapes")
for i, x in enumerate(data):
    # print("-- i:", i, x.is_contiguous(), x.is_contiguous(memory_format=torch.channels_last))
    out = cfunc(x)
    # print("cnt.frame_count:", cnt.frame_count)
    assert cnt.frame_count == expected_frame_counts[i]


torch._dynamo.reset()
cnt = torch._dynamo.testing.CompileCounter()
cfunc = torch.compile(func, backend=cnt, dynamic=True, fullgraph=True)
assert cnt.frame_count == 0
expected_frame_counts = [
    1, 2, 3, 4,
    4, 4, 4, 4,
    5, 6,
]

print("--- Check dynamic shapes")
for i, x in enumerate(data):
    # print("-- i:", i, x.is_contiguous(), x.is_contiguous(memory_format=torch.channels_last))
    out = cfunc(x)
    # print("cnt.frame_count:", cnt.frame_count)
    assert cnt.frame_count == expected_frame_counts[i]
