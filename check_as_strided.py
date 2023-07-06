# TORCH_COMPILE_DEBUG=1 python check_as_strided.py

import torch
import os


if not ("OMP_NUM_THREADS" in os.environ):
    torch.set_num_threads(1)


def func(image):
    # Set stride[0] to numel
    shape = image.shape
    numel = image.numel()
    num_channels, old_height, old_width = shape[-3:]
    strides = image.stride()
    new_strides = list(strides)
    new_strides[0] = numel
    output = image.as_strided((1, num_channels, old_height, old_width), new_strides)

    # add few other ops to make a bit more complicated graph
    output = output[:, :, 10:-10, 20:-20]
    output = output.clone()
    output = output.float()
    output = (output - 0.5) / 2.0

    return output


c_func = torch.compile(func)

x = torch.randint(0, 256, size=(1, 3, 345, 456), dtype=torch.uint8)
x = x.contiguous(memory_format=torch.channels_last)[0]
x = x.reshape(1, 3, 345, 456)

y = c_func(x)

# print(y.shape, y.stride())


