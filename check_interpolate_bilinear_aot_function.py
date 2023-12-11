import os
import torch
from torch._functorch.aot_autograd import aot_function
from torch._functorch.partitioners import min_cut_rematerialization_partition
from torch._decomp import decomposition_table, core_aten_decompositions


if not ("OMP_NUM_THREADS" in os.environ):
    torch.set_num_threads(1)


def transform(img, osize):
    # img = img[None, ...]
    img = torch.nn.functional.interpolate(img, size=osize, mode="bilinear", antialias=False)

    return img

# device = "cuda"
device = "cpu"


def nop(fx_g, _):
    return fx_g


c_transform = aot_function(
    transform,
    nop,
    nop,
    dynamic=True,
    partition_fn=min_cut_rematerialization_partition,
    decompositions=core_aten_decompositions(),
)

# memory_format = torch.channels_last
memory_format = torch.contiguous_format

# x = torch.randint(0, 256, size=(2, 3, 345, 456), dtype=torch.uint8)
# x = torch.arange(3 * 345 * 456, device=device).reshape(1, 3, 345, 456).to(torch.uint8)
# x = torch.arange(3 * 345 * 456, device=device).reshape(1, 3, 345, 456).to(torch.uint8)
# x = x.to(torch.float32)
# x = x.contiguous(memory_format=memory_format)[0]

isize = (4, 4)
osize = (3, 3)

x = torch.rand(2, 3, *isize, device=device)

output = c_transform(x, osize)
expected = transform(x, osize)
# expected_f = transform(x.float(), osize)

torch.set_printoptions(precision=6)

print(output.dtype, expected.dtype)
print(output.shape, expected.shape)
print(output.stride(), expected.stride())

# print(output[0, 0, :3, :5])
# print(expected[0, 0, :3, :5])
# print(expected_f[0, 0, :3, :5])

# m = (output.float() - expected.float()).abs() > 0
# print(output[m][:10])
# print(expected[m][:10])

torch.testing.assert_close(output, expected)
