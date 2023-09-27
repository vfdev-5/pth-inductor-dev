import torch


@torch.compile()
def fn(x):
    N, C, _, _ = x.shape
    N_idx = torch.arange(N, device=x.device).view(N, 1, 1, 1)
    C_idx = torch.arange(C, device=x.device).view(1, C, 1, 1)

    output = x[N_idx, C_idx, 0, 0] + x[N_idx, C_idx, 1, 1] + x[N_idx, C_idx, 2, 2] + x[N_idx, C_idx, 3, 3] + x[N_idx, C_idx, 4, 4] + \
        x[N_idx, C_idx, 5, 5] + x[N_idx, C_idx, 6, 6] + x[N_idx, C_idx, 7, 7] + x[N_idx, C_idx, 8, 8] + x[N_idx, C_idx, 9, 9]
    return output

x = torch.randn(8, 3, 32, 32, device="cuda")
fn(x)