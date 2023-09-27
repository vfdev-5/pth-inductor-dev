import random
import torch
import torch.utils.benchmark as benchmark

from torch.testing import make_tensor

import torch._inductor.config as inductor_config

device = 'cuda'
dtype = torch.float16
dtypeq = torch.uint8

torch.set_printoptions(precision=3, threshold=None, edgeitems=4, linewidth=460, profile=None, sci_mode=False)

@inductor_config.patch(max_autotune_gemm=True)
def gemmq_triton(a, b, c, d):
    return torch.mm(a, b) * c + d

@inductor_config.patch(force_mixed_mm=True, max_autotune_gemm=True)
def gemmq_mixed_triton(a, b, c, d):
    return torch.mm(a, b.to(a.dtype)) * c + d

gemmq_triton_compiled = torch.compile(gemmq_triton)
gemmq_mixed_triton_compiled = torch.compile(gemmq_mixed_triton)

if __name__ == '__main__':
    results = []

    min_run_time = 5

    shapes = [
        # distilbert shapes
        ###(768, 3072, 768),
        ###(3072, 768, 3072),
        # jiecao shapes
        ###(1024, 1536, 2048),
        ###(1024, 9408, 2048),
        ###(1024, 3200, 2048),
        ###(1024, 256, 9472),
        ###(1024, 10240, 256),
        ###(1024, 256, 12608),
        ###(1024, 2560, 1024),
        ###(1024, 512, 10240),
        ###(1024, 10240, 512),
        ###(1024, 2048, 1024),
        ###(1024, 512, 512),
        ###(1024, 1024, 1024),
        ###(1024, 2048, 2048),
        ###(2048, 1536, 2048),
        ###(2048, 9408, 2048),
        ###(2048, 3200, 2048),
        ###(2048, 256, 9472),
        ###(2048, 10240, 256),
        ###(2048, 256, 12608),
        ###(2048, 2560, 1024),
        ###(2048, 512, 10240),
        ###(2048, 10240, 512),
        ###(2048, 2048, 1024),
        ###(2048, 512, 512),
        ###(2048, 1024, 1024),
        (2048, 2048, 2048),
    ]

    for m, k, n in shapes:
        try:
            label = 'cuBLAS vs. Triton (same and mixed dtypes) GEMM'
            sub_label = f'm:{m:5d} | k:{k:5d} | n:{n:5d}'

            a = make_tensor(m, k, dtype=dtype, device=device)
            b = make_tensor(k, n, dtype=dtypeq, device=device)
            c = make_tensor((1, n), dtype=dtype, device=device)
            d = make_tensor((1, n), dtype=dtype, device=device)

            b_ref = b.to(dtype)
            c_ref = b_ref * c.expand((k, n))

            e0 = torch.addmm(d, a, c_ref)
            e1 = gemmq_triton_compiled(a, b_ref, c, d)
            e2 = gemmq_mixed_triton_compiled(a, b, c, d)

            ###torch.testing.assert_close(e1, e0, rtol=1e-3, atol=0)
            ###torch.testing.assert_close(e2, e0, rtol=1e-3, atol=0)

            measurement = benchmark.Timer(
                stmt='addmm(d, a, c)',
                globals={'addmm': torch.addmm, 'd': d, 'a': a, 'c': c_ref},
                label=label,
                sub_label=sub_label,
                description='cuBLAS',
            ).blocked_autorange(min_run_time=min_run_time)
            results.append(measurement)

            measurement = benchmark.Timer(
                stmt='gemmq(a, b, c, d)',
                globals={'gemmq': gemmq_triton_compiled, 'a': a, 'b': b_ref, 'c': c, 'd': d},
                label=label,
                sub_label=sub_label,
                description='Same',
            ).blocked_autorange(min_run_time=min_run_time)
            results.append(measurement)

            measurement = benchmark.Timer(
                stmt='gemmq(a, b, c, d)',
                globals={'gemmq': gemmq_mixed_triton_compiled, 'a': a, 'b': b, 'c': c, 'd': d},
                label=label,
                sub_label=sub_label,
                description='Mixed',
            ).blocked_autorange(min_run_time=min_run_time)
            results.append(measurement)

        except Exception:
            continue

    compare = benchmark.Compare(results)
    compare.colorize(rowwise=True)
    compare.print()