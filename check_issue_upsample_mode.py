# TORCH_COMPILE_DEBUG=1 python -u check_issue_upsample_nearest.py

import torch
import torch.nn.functional as F


def func(x, mode, scale):
    return F.interpolate(x, scale_factor=scale, mode=mode)


torch.manual_seed(1234321)
# cmodel = torch.compile(model, backend="eager")
cfunc = torch.compile(func, backend="inductor")

dtype = torch.float32

torch.manual_seed(1234321)
# input = torch.rand((3, 4, 32, 32), device='cuda', dtype=dtype)

input = torch.rand((3, 640, 32, 32), device='cpu', dtype=dtype)
# input = torch.ones((3, 640, 32, 32), device='cuda', dtype=dtype)
# input = torch.ones((3, 640, 32, 32), device='cpu', dtype=dtype)
# input = torch.ones((3, 640, 4, 4), device='cpu', dtype=dtype)
# input = torch.arange(3 * 640 * 4 * 4, device='cpu', dtype=dtype).reshape(3, 640, 4, 4)
# input = torch.arange(3 * 640 * 123 * 234, device='cpu', dtype=dtype).reshape(3, 640, 123, 234)

# expected = model(input.requires_grad_())
# expected.sum().backward()
# assert input.grad is not None


# output = cmodel(input.requires_grad_())
# output.sum().backward()
# torch.testing.assert_close(expected, output)

# mode = "nearest"
mode = "bicubic"
scale = 1.89


with torch.no_grad():
    expected2 = func(input, mode, scale)
with torch.no_grad():
    output2 = cfunc(input, mode, scale)

assert not expected2.requires_grad
assert not output2.requires_grad
torch.testing.assert_close(expected2, output2)


with torch.inference_mode():
    expected3 = func(input, mode, scale)

with torch.inference_mode():
    output3 = cfunc(input, mode, scale)

assert not expected3.requires_grad
assert not output3.requires_grad
torch.testing.assert_close(expected2, expected3)

print("mean expected3:", expected3.shape, expected3.mean())
print("mean output3:", output3.shape, output3.mean())

m = expected3 != output3
print("N:", m.sum())
print("expected3:", expected3[m][:10])
print("output3:", output3[m][:10])
# print(torch.argwhere(m))

torch.testing.assert_close(expected3, output3)
