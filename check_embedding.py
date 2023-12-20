import torch
import torch.nn as nn


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(64, 128)

    def forward(self, idx, x):
        return self.emb(idx)


idx = torch.randint(0, 64, (4, 32))
x = torch.randn(4, 32, 128).to(torch.bfloat16)
m = M().eval()

cm = torch.compile(m, fullgraph=True)
with torch.no_grad():

    expected = m(idx, x)
    output = cm(idx, x)

    print("expected: ", expected.shape, expected.dtype)
    print("output: ", output.shape, output.dtype)

    torch.testing.assert_close(expected, output)
