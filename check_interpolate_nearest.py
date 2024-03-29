# TORCH_COMPILE_DEBUG=1 python check_interpolate_nearest.py
# TORCH_LOGS=+output_code python check_interpolate_nearest.py

import os

import torch

if not ("OMP_NUM_THREADS" in os.environ):
    torch.set_num_threads(1)


def transform(img, osize):
    img = torch.nn.functional.interpolate(img, size=osize, mode="nearest")
    return img

# device = "cuda"
device = "cpu"

c_transform = torch.compile(transform, fullgraph=True, dynamic=True)

# memory_format = torch.channels_last
memory_format = torch.contiguous_format

# x = torch.randint(0, 256, size=(2, 3, 345, 456), dtype=torch.uint8)
# x = torch.arange(3 * 345 * 456, device=device).reshape(1, 3, 345, 456).to(torch.uint8)
# x = torch.arange(3 * 345 * 456, device=device).reshape(1, 3, 345, 456).to(torch.uint8)
# x = x.to(torch.float32)
# x = x.contiguous(memory_format=memory_format)[0]

x = torch.rand(1, 3, 501, 401, device=device)

osize = (224, 225)

output = c_transform(x, osize)
expected = transform(x, osize)
expected_f = transform(x.float(), osize)

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

torch.testing.assert_close(output, expected)
