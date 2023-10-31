import torch
import triton
from triton.testing import do_bench


def transform(img, osize):
    img = torch.nn.functional.interpolate(img, size=osize, mode="bilinear", antialias=True, align_corners=False)
    return img


def main():

    results = []
    isize = (3456, 4567)
    osize = (2345, 3456)
    bs = 4
    device = "cuda"

    torch.manual_seed(12)
    torch.set_num_threads(1)
    memory_format = torch.contiguous_format
    dtype = torch.float32

    x = torch.randint(0, 256, size=(bs, 3, *isize), dtype=dtype, device=device)
    x = x.contiguous(memory_format=memory_format)

    # c_transform = torch.compile(transform, mode="reduce-overhead")
    c_transform = torch.compile(transform)
    _ = c_transform(x, osize)
    _ = transform(x, osize)

    results.append((
        "Eager", do_bench(lambda: transform(x, osize), rep=2000, return_mode="median")
    ))
    results.append((
        "Compiled", do_bench(lambda: c_transform(x, osize), rep=2000, return_mode="median")
    ))

    print(f"Interpolate bilinear, AA=true, {device}")
    print(f"Input ({bs}, 3, {isize[0]}, {isize[1]}) -> {osize}, {x.dtype}, {memory_format}")
    for name, value in results:
        print("-", name, value)


if __name__ == "__main__":

    print("")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch config: {torch.__config__.show()}")
    print("")

    main()