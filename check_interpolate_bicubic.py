# TORCH_COMPILE_DEBUG=1 python -u check_interpolate_bicubic.py

import os

import torch

if not ("OMP_NUM_THREADS" in os.environ):
    torch.set_num_threads(1)


def transform(img, osize, align_corners):
    img = torch.nn.functional.interpolate(img, size=osize, mode="bicubic", antialias=False, align_corners=align_corners)
    return img


# backend = "eager"
# backend = "aot_eager_decomp_partition"
# backend = "aot_eager"
backend = "inductor"

c_transform = torch.compile(transform, dynamic=True, backend=backend, fullgraph=True)

# memory_format = torch.channels_last
memory_format = torch.contiguous_format
# device = "cuda"
device = "cpu"
# align_corners = False
align_corners = True

# x = torch.randint(0, 256, size=(1, 3, 270, 456), dtype=torch.uint8)
# x = torch.randint(0, 256, size=(1, 3, 345, 270), dtype=torch.uint8)

# x = torch.randint(0, 256, size=(1, 3, 345, 456), dtype=torch.uint8)
# x = torch.arange(3 * 345 * 456).reshape(1, 3, 345, 456).to(torch.uint8)
# x = torch.randint(0, 256, size=(1, 3, 400, 400), dtype=torch.uint8)

# x = torch.arange(3 * 345 * 456, device=device).reshape(1, 3, 345, 456).to(torch.uint8).to(torch.float32)
# x = torch.arange(3 * 345 * 456, device=device).reshape(1, 3, 345, 456).to(torch.uint8)

torch.manual_seed(14)
bs = 1
# x = torch.randint(0, 256, size=(bs, 3, 500, 400), dtype=torch.uint8, device=device)
# x = torch.randint(0, 256, size=(bs, 3, 1200, 1300), dtype=torch.uint8, device=device)
x = torch.randint(0, 256, size=(bs, 3, 500, 400), dtype=torch.float32, device=device)
# x = torch.arange(3 * 345 * 456, device=device).reshape(1, 3, 345, 456).to(torch.uint8).to(torch.float32)

# x = torch.arange(3 * 32 * 32, device=device).reshape(1, 3, 32, 32).to(torch.uint8).to(torch.float32)
x = x.contiguous(memory_format=memory_format)

osize = (225, 224)
# osize = (200, 300)

output = c_transform(x, osize, align_corners)
expected = transform(x, osize, align_corners)
# expected_f = transform(x.float())

torch.set_printoptions(precision=7)

print(output.dtype, expected.dtype)
print(output.shape, expected.shape)
print(output.stride(), expected.stride())

# print(output[0, 0, :, :])
# print(expected[0, 0, :, :])
# print(expected_f[0, 0, :3, :5])

# adiff = (output.float() - expected.float()).abs()
# m = adiff > 0
# print(output[m])
# print(expected[m])
# print(adiff[0, 0, ...])

torch.testing.assert_close(output, expected, atol=1e-3, rtol=0)
# torch.testing.assert_close(output[:, :, 1:-1, 1:-1], expected[:, :, 1:-1, 1:-1])
