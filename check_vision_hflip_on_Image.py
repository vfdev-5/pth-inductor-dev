# TORCH_COMPILE_DEBUG=1 python check_vision_hflip_on_Image.py

import torch
from torchvision.transforms.v2.functional import horizontal_flip
from torchvision.tv_tensors import Image


def func(x):
    x = x.flip(-1)
    # return x.as_subclass(Image)
    return Image(x)


x = torch.randint(0, 256, size=(3, 46, 52), dtype=torch.uint8)
cfn = torch.compile(func)
out = cfn(x)
print(type(out))


## HFLIP
# x = Image(torch.randint(0, 256, size=(3, 46, 52), dtype=torch.uint8))

# expected = horizontal_flip(x)
# print(expected.shape)

# cfn = torch.compile(horizontal_flip)
# out = cfn(x)
# print(out.shape)

# torch.testing.assert_close(out, expected)

# explanation = torch._dynamo.explain(horizontal_flip)(x)

# print(explanation.graph_count)
# print(explanation.graph_break_count)
# print(explanation.break_reasons)
