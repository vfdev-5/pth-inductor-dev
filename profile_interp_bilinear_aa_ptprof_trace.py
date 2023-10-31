import torch
from torch.profiler import profile, record_function, ProfilerActivity

import os


def transform(img, osize):
    img = torch.nn.functional.interpolate(img, size=osize, mode="bilinear", antialias=True, align_corners=False)

    return img


def main():

    if not ("OMP_NUM_THREADS" in os.environ):
        torch.set_num_threads(1)

    c_interp_bilinear_aa = torch.compile(transform)

    x_list = [
        torch.randint(0, 256, size=(4, 3, 3456, 4567), dtype=torch.float32, device="cuda")
        for _ in range(5)
    ]

    # warm-up
    x = x_list[0]
    y = c_interp_bilinear_aa(x, (2345, 3456))
    print(y.shape, y.dtype, y.is_contiguous())

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for x in x_list:
            out = c_interp_bilinear_aa(x, (2345, 3456))
            prof.step()
            out.sum()

    name = "interp_bilinear_aa"
    index = 0
    filename = f"compiled_{name}_cuda_trace{index}.json"
    while os.path.exists(filename):
        index += 1
        filename = f"compiled_{name}_cuda_trace{index}.json"
    prof.export_chrome_trace(filename)

    x = x_list[0]
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with record_function(f"c_{name}"):
            c_interp_bilinear_aa(x, (2345, 3456))

    index = 0
    filename = f"trace_compile_{name}_cuda{index}.json"
    while os.path.exists(filename):
        index += 1
        filename = f"trace_compile_{name}_cuda{index}.json"
    prof.export_chrome_trace(filename)


if __name__ == "__main__":
    main()
