# TORCH_COMPILE_DEBUG=1 python check_vflip.py
# TORCH_LOGS=+inductor python check_vflip.py

import torch
import os


if not ("OMP_NUM_THREADS" in os.environ):
    torch.set_num_threads(1)


def n_flip(x, dim):
    o = torch.flip(x, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))

    o = torch.flip(o, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))

    o = torch.flip(o, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))

    o = torch.flip(o, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))

    o = torch.flip(o, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))

    o = torch.flip(o, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))

    o = torch.flip(o, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))

    o = torch.flip(o, dims=(dim, ))
    o = torch.flip(o, dims=(dim, ))

    o = torch.flip(o, dims=(dim, ))
    return o


n_flip_inductor = torch.compile(n_flip)

x = torch.randint(0, 256, size=(1, 3, 224, 224), dtype=torch.uint8)

y = n_flip_inductor(x, dim=-2)

print(y.shape, y.dtype, y.is_contiguous())
torch.testing.assert_close(y, x.flip(dims=(-2, )))



