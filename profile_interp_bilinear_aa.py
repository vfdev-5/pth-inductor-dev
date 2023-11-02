import os
import torch
from torch.profiler import profile, record_function, ProfilerActivity


def transform(img, osize):
    img = torch.nn.functional.interpolate(img, size=osize, mode="bilinear", antialias=True, align_corners=False)
    return img


def main():

    if not ("OMP_NUM_THREADS" in os.environ):
        torch.set_num_threads(1)

    c_interp_bilinear_aa = torch.compile(transform)
    # c_interp_bilinear_aa = transform

    # isize = (3456, 4567)
    # osize = (2345, 3456)

    n = 4
    isize = (500, 400)
    osize = (256, 256)

    x = torch.randint(0, 256, size=(n, 3, *isize), dtype=torch.float32, device="cuda")

    # warm-up
    y = c_interp_bilinear_aa(x, osize)
    print(y.shape, y.dtype, y.is_contiguous())

    for _ in range(10):
        _ = c_interp_bilinear_aa(x, osize)

    torch.cuda.synchronize()

    n = 1000

    with profile(
        activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA
        ],
        record_shapes=True
    ) as prof:
        with record_function("loop_interp_bilinear_aa_cuda"):
            for _ in range(n):
                c_interp_bilinear_aa(x, osize)

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    call_fn = None
    from torch._inductor.codecache import PyCodeCache

    for key in PyCodeCache.cache:
        mod = PyCodeCache.cache[key]
        if "call" in mod.__dict__:
            call_fn = mod.__dict__["call"]

    for _ in range(10):
        _ = call_fn([x])

    with profile(
        activities=[
            ProfilerActivity.CPU, ProfilerActivity.CUDA
        ],
        record_shapes=True
    ) as prof:
        with record_function("loop_triton"):
            for _ in range(n):
                call_fn([x])

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


if __name__ == "__main__":
    main()
