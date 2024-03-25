# TORCH_COMPILE_DEBUG=1 python check_interpolate_bilinear.py
# TORCH_LOGS=+output_code python check_interpolate_bilinear.py

import os

import torch

if not ("OMP_NUM_THREADS" in os.environ):
    torch.set_num_threads(1)


def transform(img, osize, align_corners):
    img = torch.nn.functional.interpolate(img, size=osize, mode="bilinear", antialias=False, align_corners=align_corners)
    return img

# device = "cuda"
device = "cpu"
align_corners = False

# backend = "eager"
# backend = "aot_eager_decomp_partition"
# backend = "aot_eager"
backend = "inductor"

c_transform = torch.compile(transform, dynamic=True, backend=backend, fullgraph=True)

# memory_format = torch.channels_last
memory_format = torch.contiguous_format


#   Input (4, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: True, antialias: False, osize: (200, 300)     |       9866.441 (+-50.617)       |        5639.161 (+-16.663)         |           1628.572 (+-19.770)           |     0.289 (+-0.000)      |         9893.606 (+-62.377)

torch.manual_seed(12)
# x = torch.randint(0, 256, size=(2, 3, 500, 400), dtype=torch.uint8)
# x = torch.randint(0, 256, size=(1, 3, 500, 400), dtype=torch.uint8)
# x = torch.randint(0, 256, size=(4, 3, 1200, 1300), dtype=torch.uint8)
# x = torch.randint(0, 256, size=(1, 3, 500, 400), dtype=torch.float32)

# Input (1, 3, 1200, 1300), torch.uint8, torch.contiguous_format | mode: bilinear, align_corners: True, osize: (200, 300)
# x = torch.randint(0, 256, size=(1, 3, 1200, 1300), dtype=torch.uint8, device=device)

x = torch.randint(0, 256, size=(1, 3, 2345, 2456), dtype=torch.float32, device=device)

# x = torch.arange(3 * 345 * 456, device=device).reshape(1, 3, 345, 456).to(torch.uint8, device=device)
# x = torch.arange(3 * 345 * 456, device=device).reshape(1, 3, 345, 456).to(torch.uint8, device=device)
# x = x.to(torch.float32)
x = x.contiguous(memory_format=memory_format)
x = x.to(device=device)

# isize = (4, 4)
# osize = (3, 3)
# x = torch.rand(2, 3, *isize, device=device)

# osize = (500, 256)
# osize = (300, 256)
# osize = (200, 300)
# osize = (600, 700)
osize = (1234, 1345)

output = c_transform(x, osize, align_corners=align_corners)
expected = transform(x, osize, align_corners=align_corners)
# expected_f = transform(x.float(), osize)

torch.set_printoptions(precision=6)

print(output.dtype, expected.dtype)
print(output.shape, expected.shape)
print(output.stride(), expected.stride())

print(output[0, 0, :3, :5])
print(expected[0, 0, :3, :5])
# print(expected_f[0, 0, :3, :5])

# m = (output.float() - expected.float()).abs() > 0
# print(output[m][:10])
# print(expected[m][:10])

kwargs = {}
if x.dtype == torch.uint8:
    kwargs = {
        "atol": 1.0,
        "rtol": 0.0,
    }

# torch.testing.assert_close(expected.float(), expected_f, **kwargs)
torch.testing.assert_close(output, expected, **kwargs)

