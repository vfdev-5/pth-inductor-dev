
import torch
import torchvision

torchvision.disable_beta_transforms_warning()

from torchvision.transforms.v2 import functional as F


class MyModule(torch.nn.Module):
    def forward(self, x):
        o = F.resize(x, size=(12, 12), antialias=False)
        o = F.center_crop(x, 5)
        return o.sum()

module = MyModule()


import torch
import torch._dynamo as dynamo

explanation = dynamo.explain(module, torch.randint(0, 256, size=(1, 3, 32, 32), dtype=torch.uint8))
print(
    explanation.graph_count,
    explanation.graph_break_count,
    explanation.break_reasons,
)
