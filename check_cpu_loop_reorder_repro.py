# TORCH_COMPILE_DEBUG=1 python check_cpu_loop_reorder_repro.py
# TORCH_LOGS=+inductor python check_cpu_loop_reorder_repro.py

import torch
import os


if not ("OMP_NUM_THREADS" in os.environ):
    torch.set_num_threads(1)


def func(x):
    x = x.flip(-2)
    x += 12.3
    return x


cfunc = torch.compile(func)

x = torch.randint(0, 256, size=(1, 3, 2345, 3456), dtype=torch.uint8).contiguous(memory_format=torch.channels_last)
# x = torch.randint(0, 256, size=(1, 3, 224, 224), dtype=torch.uint8)

x = x.to(torch.float32)

# x = torch.randint(0, 256, size=(1, 3, 224 * 2, 224 * 2), dtype=torch.uint8)
# x = x[:, :, ::2, ::2]

# x = torch.randint(0, 256, size=(1, 3, 224 + 21, 224 + 22), dtype=torch.uint8)
# x = x[:, :, 9:-12, 12:-10]

y = cfunc(x)

print(y.shape, y.dtype, y.is_contiguous())
