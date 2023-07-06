# TORCH_COMPILE_DEBUG=1 python check_vflip.py
# TORCH_LOGS=+inductor python check_vflip.py

import torch
import os


if not ("OMP_NUM_THREADS" in os.environ):
    torch.set_num_threads(1)


def vflip_uint8_rgb(x, dim):
    return x.flip(dims=(dim, ))


c_vflip_uint8_rgb = torch.compile(vflip_uint8_rgb)

# x = torch.randint(0, 256, size=(1, 3, 224, 224), dtype=torch.uint8).contiguous(memory_format=torch.channels_last)
x = torch.randint(0, 256, size=(1, 3, 224, 224), dtype=torch.uint8)

# x = torch.randint(0, 256, size=(1, 3, 224 * 2, 224 * 2), dtype=torch.uint8)
# x = x[:, :, ::2, ::2]

# x = torch.randint(0, 256, size=(1, 3, 224 + 21, 224 + 22), dtype=torch.uint8)
# x = x[:, :, 9:-12, 12:-10]

y = c_vflip_uint8_rgb(x, dim=-2)

print(y.shape, y.dtype, y.is_contiguous())
torch.testing.assert_close(y, x.flip(dims=(-2, )))



