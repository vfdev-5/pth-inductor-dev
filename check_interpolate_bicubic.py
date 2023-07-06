# TORCH_COMPILE_DEBUG=1 python check_interpolate_bicubic.py

import os

import torch

if not ("OMP_NUM_THREADS" in os.environ):
    torch.set_num_threads(1)


def transform(img):
    # img = img[None, ...]
    img = torch.nn.functional.interpolate(img, size=(270, 270), mode="bicubic", antialias=False)
    # img = torch.nn.functional.interpolate(img, size=(12, 32), mode="bicubic", antialias=False)

    return img

c_transform = torch.compile(transform)

# memory_format = torch.channels_last
memory_format = torch.contiguous_format
# device = "cuda"
device = "cpu"


# x = torch.randint(0, 256, size=(1, 3, 270, 456), dtype=torch.uint8)
# x = torch.randint(0, 256, size=(1, 3, 345, 270), dtype=torch.uint8)

# x = torch.randint(0, 256, size=(1, 3, 345, 456), dtype=torch.uint8)
# x = torch.arange(3 * 345 * 456).reshape(1, 3, 345, 456).to(torch.uint8)
x = torch.randint(0, 256, size=(1, 3, 400, 400), dtype=torch.uint8)
# x = torch.arange(3 * 345 * 456, device=device).reshape(1, 3, 345, 456).to(torch.uint8).to(torch.float32)
# x = torch.arange(3 * 32 * 32, device=device).reshape(1, 3, 32, 32).to(torch.uint8).to(torch.float32)
# x = x.contiguous(memory_format=memory_format)[0]

output = c_transform(x)
expected = transform(x)
# expected_f = transform(x.float())

torch.set_printoptions(precision=7)

print(output.dtype, expected.dtype)
print(output.shape, expected.shape)
print(output.stride(), expected.stride())

# print(output[0, 0, :, :])
# print(expected[0, 0, :, :])
# print(expected_f[0, 0, :3, :5])


# (output.float() - expected.float()).abs()

adiff = (output.float() - expected.float()).abs()
m = adiff > 0
# print(output[m])
# print(expected[m])
# print(adiff[0, 0, ...])

torch.testing.assert_close(output, expected)
# torch.testing.assert_close(output[:, :, 1:-1, 1:-1], expected[:, :, 1:-1, 1:-1])
