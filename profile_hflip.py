import torch
import os


def main():

    if not ("OMP_NUM_THREADS" in os.environ):
        torch.set_num_threads(1)


    def hflip_uint8_rgb(x, dim):
        return x.flip(dims=(dim, ))

    c_hflip_uint8_rgb = torch.compile(hflip_uint8_rgb)

    x = torch.randint(0, 256, size=(1, 3, 224, 224), dtype=torch.uint8, device="cuda")

    y = c_hflip_uint8_rgb(x, dim=-1)
    torch.cuda.synchronize()
    print(y.shape, y.dtype, y.is_contiguous())
    torch.testing.assert_close(y, x.flip(dims=(-1, )))

    # Run cProfile

    for _ in range(10):
        _ = c_hflip_uint8_rgb(x, dim=-1)
        torch.cuda.synchronize()

    import cProfile, io, pstats

    n = 10000

    prof_filename = None
    filename = "profile_hflip.log"
    # if isinstance(filename, str) and filename.endswith(".prof"):
    #     prof_filename = filename

    with cProfile.Profile(timeunit=0.000001) as pr:
        for i in range(n):
            _ = c_hflip_uint8_rgb(x, dim=-1)
            torch.cuda.synchronize()

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
