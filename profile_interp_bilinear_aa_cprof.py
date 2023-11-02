import torch
import os


def transform(img, osize):
    img = torch.nn.functional.interpolate(img, size=osize, mode="bilinear", antialias=True, align_corners=False)
    return img


def main():

    if not ("OMP_NUM_THREADS" in os.environ):
        torch.set_num_threads(1)

    c_interp_bilinear_aa = torch.compile(transform)
    # c_interp_bilinear_aa = transform

    x = torch.randint(0, 256, size=(4, 3, 3456, 4567), dtype=torch.float32, device="cuda")

    # warm-up
    y = c_interp_bilinear_aa(x, (2345, 3456))
    print(y.shape, y.dtype, y.is_contiguous())

    for _ in range(10):
        _ = c_interp_bilinear_aa(x, (2345, 3456))

    torch.cuda.synchronize()

    import cProfile, pstats

    n = 1000

    index = 0
    filename = f"profile_interp_bilinear_aa_cuda_{index}.log"
    while os.path.exists(filename):
        index += 1
        filename = f"profile_interp_bilinear_aa_cuda_{index}.log"

    with cProfile.Profile(timeunit=0.000001) as pr:
        for _ in range(n):
            out = c_interp_bilinear_aa(x, (2345, 3456))
            torch.cuda.synchronize()
            # pr.disable()
            # assert out[0, 0, 0, 0].item() > -1
            # pr.enable()

    with open(filename, "w") as h:
        ps = pstats.Stats(pr, stream=h).sort_stats("tottime")
        ps.print_stats()


if __name__ == "__main__":
    main()
