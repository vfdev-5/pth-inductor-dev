import torch


def func(x):
    if x.shape[0] > 2:
        return x + 2
    else:
        return x * 2

x = torch.rand(10)

graph, guards = torch._dynamo.export(func)(x)
print("\n--- graph:")
graph.print_readable()

print("\n--- guards:")

guard_code = []
for guard in guards:
    if guard.code_list:
        guard_code += guard.code_list

print("\n".join(guard_code))


##############################

# import torch
# import torch._dynamo.testing

# cnt = torch._dynamo.testing.CompileCounter()

# @torch._dynamo.optimize(cnt)
# def func(x):
#     if x.shape[0] > 2:
#         return x + 2
#     else:
#         return x * 2

# func(torch.randn(5))
# func(torch.randn(6))
# func(torch.randn(7))
# func(torch.randn(1))

# print(cnt.frame_count)
# print(cnt.op_count)
