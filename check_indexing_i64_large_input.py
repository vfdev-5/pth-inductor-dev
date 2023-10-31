# TORCH_COMPILE_DEBUG=1 python check_indexing_i64_large_input.py
import torch


def transform(x, indices):
    return x[indices]



size = 2 ** 32
x = torch.rand(size, dtype=torch.float32, device="cuda")
indices = [size - 1, 0, 10, size - 2, 100]

expected = transform(x, indices)

c_transform = torch.compile(transform)

output = c_transform(x, indices)

torch.testing.assert_close(expected, output)
