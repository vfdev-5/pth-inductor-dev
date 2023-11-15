# TORCH_COMPILE_DEBUG=1 python check_interpolate_bilinear_aa.py

import os
import torch

# We need to set cache size very large to avoid benchmarking eager mode as compiled
torch._dynamo.config.cache_size_limit = 100000


if not ("OMP_NUM_THREADS" in os.environ):
    torch.set_num_threads(1)


run_pass = "both"


def transform_hp(img, osize):
    h = img.shape[-2]
    img = torch.nn.functional.interpolate(img, size=(h, osize[1]), mode="bilinear", antialias=True)
    return img


def transform_vp(img, osize):
    w = img.shape[-1]
    img = torch.nn.functional.interpolate(img, size=(osize[0], w), mode="bilinear", antialias=True)
    return img


def transform_both(img, osize):
    img = torch.nn.functional.interpolate(img, size=osize, mode="bilinear", antialias=True)
    return img


tr_map = {
    "both": transform_both,
    "hp": transform_hp,
    "vp": transform_vp,
}

isize, osize = (500, 400), (256, 256)
# isize, osize = (500, 400), (500 // 5, 400 // 5)
# isize, osize = (3456, 4567), (2345, 3456)

transform = tr_map[run_pass]

# device = "cuda"
device = "cpu"

c_transform = torch.compile(transform)

# memory_format = torch.channels_last
memory_format = torch.contiguous_format

# for n in [1, 4]:
for n in [4, ]:
# for n in [1, ]:
    # x = torch.randint(0, 256, size=(n, 3, *isize), dtype=torch.uint8)
    # x = torch.arange(3 * isize[0] * isize[1], device=device).reshape(1, 3, *isize).to(torch.uint8)

    # x = torch.arange(3 * isize[0] * isize[1], device=device).reshape(1, 3, *isize).to(torch.uint8)
    # x = x.to(torch.float32)

    x = torch.rand(n, 3, *isize, device=device)
    x = x.contiguous(memory_format=memory_format)

    output = c_transform(x, osize)
    expected = transform(x, osize)
    expected_f = transform(x.float(), osize)

    # for _ in range(10):
    #     _ = c_transform(x, osize)
    # _ = c_transform(x, osize)

    from torch._inductor.codecache import PyCodeCache

    for key in PyCodeCache.cache:
        print(key, PyCodeCache.cache[key].__file__)

    if x.is_floating_point():
        torch.testing.assert_close(output, expected)
    else:
        torch.testing.assert_close(output.float(), expected_f, atol=1.0, rtol=0.0)

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


# ## Check backward pass

# x = torch.rand(2, 3, 345, 456, device=device)
# x1 = x.clone()
# x1.requires_grad_(True)
# x2 = x.clone()
# x2.requires_grad_(True)

# output = c_transform(x1)
# expected = transform(x2)

# output.sum().backward()
# expected.sum().backward()

# torch.testing.assert_close(x1.grad, x2.grad)
