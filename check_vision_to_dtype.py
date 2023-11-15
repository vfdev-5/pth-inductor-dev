# TORCH_COMPILE_DEBUG=1 python check_vision_to_dtype.py

import torch
from torchvision.transforms.v2.functional import to_dtype


x = torch.randint(0, 256, size=(3, 46, 52), dtype=torch.uint8)

expected = to_dtype(x, dtype=torch.float32)
print(expected.dtype)

cfn = torch.compile(to_dtype)
out = cfn(x)
print(out.dtype)

torch.testing.assert_close(out, expected)

explanation = torch._dynamo.explain(to_dtype)(x)

print(explanation.graph_count)
print(explanation.graph_break_count)
print(explanation.break_reasons)
