# # TORCH_COMPILE_DEBUG=1 python check_vision_equalize.py

import torch
from torchvision.transforms.v2.functional import equalize, equalize_image


x = torch.randint(0, 256, size=(3, 46, 52), dtype=torch.uint8)

expected = equalize_image(x)
print(expected.shape)

cfn = torch.compile(equalize_image)
out = cfn(x)
print(out.shape)

torch.testing.assert_close(out, expected)


explanation = torch._dynamo.explain(equalize)(x)

print(explanation.graph_count)
print(explanation.graph_break_count)
print(explanation.break_reasons)




# # lut = torch.cat([lut.new_zeros(1).expand(batch_shape + (1,)), lut], dim=-1)

# def func(x):
#     batch_shape = x.shape[:1]
#     out = torch.cat([x.new_zeros(1).expand(batch_shape + (1,)), x], dim=-1)
#     return out


# cfunc = torch.compile(func)

# x = torch.randint(0, 256, size=(3, 255), dtype=torch.float32)
# expected = func(x)
# out = cfunc(x)
# print("1", expected.shape, out.shape)


# x = torch.randint(0, 256, size=(3, 255), dtype=torch.uint8)
# expected = func(x)
# out = cfunc(x)
# print("2", expected.shape, out.shape)