# TORCH_COMPILE_DEBUG=1 python check_interpolate_bilinear_aa.py

import os

import torch

if not ("OMP_NUM_THREADS" in os.environ):
    torch.set_num_threads(1)


def transform(img):
    # img = torch.nn.functional.interpolate(img, size=(271, 272), mode="bilinear", antialias=True)
    img = torch.nn.functional.interpolate(img, size=(345, 272), mode="bilinear", antialias=True)

    return img

# device = "cuda"
device = "cpu"

c_transform = torch.compile(transform)

memory_format = torch.channels_last
# memory_format = torch.contiguous_format

# for n in [1, 4]:
for n in [1, ]:
    # x = torch.randint(0, 256, size=(n, 3, 345, 456), dtype=torch.uint8)
    # x = torch.arange(3 * 345 * 456, device=device).reshape(1, 3, 345, 456).to(torch.uint8)

    # x = torch.arange(3 * 345 * 456, device=device).reshape(1, 3, 345, 456).to(torch.uint8)
    # x = x.to(torch.float32)

    x = torch.rand(n, 3, 345, 456, device=device)
    x = x.contiguous(memory_format=memory_format)

    output = c_transform(x)
    expected = transform(x)
    expected_f = transform(x.float())

    if x.is_floating_point():
        torch.testing.assert_close(output, expected)
    else:
        torch.testing.assert_close(output.float(), expected_f, atol=1.0, rtol=0.0)

torch.set_printoptions(precision=6)

print(output.dtype, expected.dtype)
print(output.shape, expected.shape)
print(output.stride(), expected.stride())

print(output[0, 0, :3, :5])
print(expected[0, 0, :3, :5])
print(expected_f[0, 0, :3, :5])

# m = (output.float() - expected.float()).abs() > 0
# print(output[m][:10])
# print(expected[m][:10])


# ## Check backward pass

# x = torch.rand(2, 3, 345, 456, device=device)
# x1 = x.clone()
# x1.requires_grad_(True)
# x2 = x.clone()
# x2.requires_grad_(True)

# output = c_transform(x1)
# expected = transform(x2)

# output.sum().backward()
# expected.sum().backward()

# torch.testing.assert_close(x1.grad, x2.grad)
