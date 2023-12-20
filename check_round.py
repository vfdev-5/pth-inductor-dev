# # TORCH_COMPILE_DEBUG=1 python -u check_round.py

import torch
import torch._dynamo.testing


def func(x, a):
    y = x + round(a / 2)
    return y


torch._dynamo.reset()
cnt = torch._dynamo.testing.CompileCounter()
# cnt = "inductor"
cfunc = torch.compile(func, backend=cnt, dynamic=True, fullgraph=True)

device = "cpu"  # or "cuda"
# device = "cuda"
x = torch.tensor([0, ], dtype=torch.float32, device=device)

assert cnt.frame_count == 0
out1 = cfunc(x, 13)

print(cnt.frame_count, out1)
assert cnt.frame_count == 1

out2 = cfunc(x, 15)
print(cnt.frame_count, out2)
assert cnt.frame_count == 1

torch.testing.assert_close(out1, func(x, 13))
torch.testing.assert_close(out2, func(x, 15))
