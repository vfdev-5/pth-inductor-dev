# TORCH_COMPILE_DEBUG=1 python check_vflip.py
# TORCH_LOGS=+inductor python check_vflip.py

import torch
import os


def main():

    if not ("OMP_NUM_THREADS" in os.environ):
        torch.set_num_threads(1)


    def vflip_uint8_rgb(x, dim):
        return x.flip(dims=(dim, ))


    c_vflip_uint8_rgb = torch.compile(vflip_uint8_rgb)

    x = torch.randint(0, 256, size=(1, 3, 224, 224), dtype=torch.uint8)

    y = c_vflip_uint8_rgb(x, dim=-2)
    print(y.shape, y.dtype, y.is_contiguous())
    torch.testing.assert_close(y, x.flip(dims=(-2, )))

    # Run cProfile

    for _ in range(10):
        _ = c_vflip_uint8_rgb(x, dim=-2)

    import cProfile, io, pstats

    n = 10000

    prof_filename = None
    filename = "profile_vflip.log"
    # if isinstance(filename, str) and filename.endswith(".prof"):
    #     prof_filename = filename

    with cProfile.Profile(timeunit=0.000001) as pr:
        for i in range(n):
            _ = c_vflip_uint8_rgb(x, dim=-2)

    if filename is None:
        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("tottime")
        ps.print_stats()
        print(s.getvalue())
    elif prof_filename is not None:
        pr.dump_stats(prof_filename)
    else:
        with open(filename, "w") as h:
            ps = pstats.Stats(pr, stream=h).sort_stats("tottime")
            ps.print_stats()


if __name__ == "__main__":
    main()
