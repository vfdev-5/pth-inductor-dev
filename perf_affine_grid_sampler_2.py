
# Taken from https://github.com/pytorch/pytorch/issues/104296#issuecomment-1613763150
import torch

from torch.utils.benchmark import Timer, Compare

current = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def torch_affine(x, affine):
    grid = torch.nn.functional.affine_grid(affine, x.shape, align_corners=True)
    yt = torch.nn.functional.grid_sample(x, grid, mode="nearest", align_corners=True)
    return yt, grid


def time_forward(x, affine, compile=False):
    t = Timer(stmt="fn(x, affine)",
              globals={
                  "x": x,
                  "affine": affine,
                  "fn": torch.compile(torch_affine) if compile else torch_affine
              },
              description="Compiled" if compile else "Eager")
    res = t.blocked_autorange(min_run_time=2.)
    return res


def warp_perf(spatial_size):
    n, ch = 16, 32  # batch and channels
    x = torch.eye(spatial_size, dtype=torch.float32).repeat(n, ch, 1, 1).to(current)
    phi = torch.rand(n) * 3.141592
    s, c = torch.sin(phi), torch.cos(phi)
    r1 = torch.stack([c, -s, torch.zeros(n)], 1)
    r2 = torch.stack([s, c, torch.zeros(n)], 1)
    affine = torch.stack([r1, r2], 1).to(current)
    benchmarks = []
    benchmarks.append(time_forward(x, affine, compile=False))
    benchmarks.append(time_forward(x, affine, compile=True))
    compare = Compare(benchmarks)
    compare.print()


if __name__ == "__main__":
    warp_perf(512)
