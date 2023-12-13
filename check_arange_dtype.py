# TORCH_COMPILE_DEBUG=1 python check_arange_dtype.py
# TORCH_LOGS=+output_code python check_arange_dtype.py
import torch


def func_a(t, out_size):
    _, _, h, w = t.shape

    j = torch.arange(out_size[0], device=t.device, dtype=torch.float32)
    y_f32 = (h / out_size[0]) * j
    y_f32 = y_f32.unsqueeze(-1)
    y = y_f32.to(torch.long)

    i = torch.arange(out_size[1], device=t.device, dtype=torch.float32)
    x_f32 = (w / out_size[1]) * i
    x = x_f32.to(torch.long)

    output = t[..., y, x]
    return output


def func_b(t, out_size):
    _, _, h, w = t.shape

    j = torch.arange(out_size[0], device=t.device).to(dtype=torch.float32)
    y_f32 = (h / out_size[0]) * j
    y_f32 = y_f32.unsqueeze(-1)
    y = y_f32.to(torch.long)

    i = torch.arange(out_size[1], device=t.device).to(dtype=torch.float32)
    x_f32 = (w / out_size[1]) * i
    x = x_f32.to(torch.long)

    output = t[..., y, x]
    return output


cfunc_a = torch.compile(func_a, dynamic=True, fullgraph=True)
cfunc_b = torch.compile(func_b, dynamic=True, fullgraph=True)


device = "cuda"
isize = (20, 30)
a = torch.arange(2 * 3 * isize[0] * isize[1], device=device).reshape(2, 3, *isize)

out_size = (11, 12)

output_a = cfunc_a(a, out_size)
output_b = cfunc_b(a, out_size)

torch.testing.assert_close(output_a, output_b)
