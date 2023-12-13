# TORCH_COMPILE_DEBUG=1 python check_dyn_float_scale_from_size.py
import torch


def func(t, out_size):
    _, in_size = t.shape
    scale = (out_size - 1.0) / (in_size - 1.0)
    i = torch.arange(out_size, device=t.device)
    # x_f32 = scale * (i + 0.0)
    x_f32 = scale * i
    x = x_f32.to(torch.long)
    output = t[..., x]
    return output


cfunc = torch.compile(func, dynamic=True, fullgraph=True)

isize = 20
a = torch.arange(2 * isize).reshape(2, isize)

osize = 14
expected = func(a, osize)
output = cfunc(a, osize)

torch.testing.assert_close(expected, output)
