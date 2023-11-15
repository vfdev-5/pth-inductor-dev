# TORCH_COMPILE_DEBUG=1 python check_vision_perspective.py

import torch
from torchvision.transforms.v2.functional import perspective
from torchvision.tv_tensors import BoundingBoxes, Image


# x = torch.randint(0, 256, size=(2, 3, 500, 400), dtype=torch.uint8).to(memory_format=torch.channels_last, copy=True)
# x = BoundingBoxes(torch.randint(0, 256, size=(5, 4), dtype=torch.uint8), format="XYXY", canvas_size=(256, 256))
# x = Image(torch.randint(0, 256, size=(3, 46, 52), dtype=torch.uint8))

x = torch.randint(0, 256, size=(1, 3, 500, 400), dtype=torch.float32)
x = x[0][None, ...]

# expected = perspective(
#     x,
#     [[0, 0], [400, 0], [400, 500], [0, 500]],
#     [[10, 20], [350, 30], [420, 440], [-5, 480]],
# )
# print(expected.shape)

cfn = torch.compile(perspective)
out = cfn(
    x,
    [[0, 0], [400, 0], [400, 500], [0, 500]],
    [[10, 20], [350, 30], [420, 440], [-5, 480]],
)
print(out.shape)

# torch.testing.assert_close(out, expected)

explanation = torch._dynamo.explain(perspective)(
    x,
    [[0, 0], [400, 0], [400, 500], [0, 500]],
    [[10, 20], [350, 30], [420, 440], [-5, 480]],
)

print(explanation.graph_count)
print(explanation.graph_break_count)
print(explanation.break_reasons)
