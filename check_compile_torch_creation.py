# TORCH_COMPILE_DEBUG=1 python check_compile_torch_creation.py

import torch
import os


if not ("OMP_NUM_THREADS" in os.environ):
    torch.set_num_threads(1)


def func(n, w, h, device):
    dtype = torch.float32
    grid_x = torch.linspace(-1, 1, steps=w, dtype=dtype, device=device).view(1, 1, w, 1).expand(n, h, w, 1)
    grid_y = torch.linspace(-1, 1, steps=h, dtype=dtype, device=device).view(1, h, 1).expand(n, h, w, 1)
    grid_one = torch.tensor(1, dtype=dtype, device=device).view(1, 1, 1, 1).expand(n, h, w, 1)

    grid_x = torch.nn.functional.pad(grid_x, pad=(0, 2), mode="constant", value=0)
    grid_y = torch.nn.functional.pad(grid_y, pad=(1, 1), mode="constant", value=0)
    grid_one = torch.nn.functional.pad(grid_one, pad=(2, 0), mode="constant", value=0)

    return grid_x + grid_y + grid_one

    # return torch.stack([grid_x, grid_y, grid_one], dim=-1)


c_func = torch.compile(func)

y = c_func(1, 32, 32, "cuda")



