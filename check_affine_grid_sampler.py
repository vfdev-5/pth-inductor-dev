import torch

from torch.nn.functional import grid_sample, affine_grid

print("")
print(f"Torch version: {torch.__version__}")
print(f"Torch config: {torch.__config__.show()}")
print("")

torch.set_printoptions(precision=7)

def transform(img, theta):
    n, c, h, w = img.shape
    grid = affine_grid(theta, size=(n, c, h, w), align_corners=False)

    # print("grid x:", grid[0, 0, :20, 0])
    # print("grid y:", grid[0, :20, 0, 1])

    # # check src indices from grid:
    # gx = grid[..., 0]
    # gy = grid[..., 1]

    # # srcx = (gx + 1.0) * w * 0.5 - 0.5
    # # srcy = (gy + 1.0) * h * 0.5 - 0.5

    # srcx = gx * w * 0.5 + w * 0.5 - 0.5
    # srcy = gy * h * 0.5 + h * 0.5 - 0.5

    # print("src x:", srcx[0, 0, :20])
    # print("src y:", srcy[0, :20, 0])

    output = grid_sample(img, grid, align_corners=False)
    return output


a = torch.deg2rad(torch.tensor(45.0))
s1 = 1.23
s2 = 1.34
# a = torch.deg2rad(torch.tensor(0.0))
# s1 = 1.0
# s2 = 1.0
ca, sa = torch.cos(a), torch.sin(a)

device = "cpu"
# device = "cuda"

torch.manual_seed(12)
num_threads = 1
torch.set_num_threads(num_threads)

# memory_format = torch.contiguous_format
memory_format = torch.channels_last
# dtype = torch.float64
dtype = torch.float32


x = torch.arange(3 * 345 * 456, device=device).reshape(1, 3, 345, 456).to(torch.uint8)

# x = torch.arange(3 * 32 * 32, device=device).reshape(1, 3, 32, 32).to(torch.uint8)  # pass
# x = torch.arange(3 * 33 * 34, device=device).reshape(1, 3, 33, 34).to(torch.uint8)  # fail
# x = torch.arange(3 * 34 * 34, device=device).reshape(1, 3, 34, 34).to(torch.uint8)  # fail
# x = torch.arange(3 * 64 * 64, device=device).reshape(1, 3, 64, 64).to(torch.uint8)  # fail
# x = torch.arange(3 * 128 * 128, device=device).reshape(1, 3, 128, 128).to(torch.uint8)  # fail

x = x.to(dtype=dtype)
x = x.contiguous(memory_format=memory_format)


theta = torch.tensor(
    [[
        [ca / s1, sa, 0.0],
        [-sa, ca / s2, 0.0],
    ]],
    device=device,
    dtype=x.dtype
)

c_transform = torch.compile(transform)

output = c_transform(x, theta)
expected = transform(x, theta)


adiff = (output.float() - expected.float()).abs()
m = adiff > 1e-3

print("adiff:", adiff[m][:7])
print("output vs expected:", [
    (a.item(), b.item()) for a, b in zip(output[m][:7], expected[m][:7])
])

torch.testing.assert_close(output, expected)
