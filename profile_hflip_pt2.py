# TORCH_COMPILE_DEBUG=1 python check_hflip.py
# TORCH_LOGS=+inductor python check_hflip.py

import torch
import os


def main():

    if not ("OMP_NUM_THREADS" in os.environ):
        torch.set_num_threads(1)


    def hflip_uint8_rgb(x, dim):
        return x.flip(dims=(dim, ))


    c_hflip_uint8_rgb = torch.compile(hflip_uint8_rgb, mode="reduce-overhead")

    x_list = [
        torch.randint(0, 256, size=(1, 3, 224, 224), dtype=torch.uint8, device="cuda")
        for _ in range(20)
    ]

    # warm-up
    x = x_list[0]
    y = c_hflip_uint8_rgb(x, dim=-1)
    print(y.shape, y.dtype, y.is_contiguous())
    torch.testing.assert_close(y, x.flip(dims=(-1, )))

    with torch.profiler.profile() as prof:
        for x in x_list:
            c_hflip_uint8_rgb(x, dim=-1)
            prof.step()

    index = 0
    filename = f"compiled_hflip_cuda_trace{index}.json"
    while os.path.exists(filename):
        index += 1
        filename = f"compiled_hflip_cuda_trace{index}.json"
    prof.export_chrome_trace(filename)

    x = x_list[0]
    with torch.profiler.profile() as prof:
        with torch.profiler.record_function("c_hflip_uint8_rgb"):
            c_hflip_uint8_rgb(x, dim=-1)

    index = 0
    filename = f"trace_compile_hflip_cuda{index}.json"
    while os.path.exists(filename):
        index += 1
        filename = f"trace_compile_hflip_cuda{index}.json"
    prof.export_chrome_trace(filename)


if __name__ == "__main__":
    main()
