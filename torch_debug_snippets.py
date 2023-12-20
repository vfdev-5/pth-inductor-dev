import torch
import torch._dynamo.testing

# We need to set cache size very large to avoid running eager mode as compiled
torch._dynamo.config.cache_size_limit = 100000

# def func(x):
#     return x.sum()

# torch._dynamo.reset()
# cnt = torch._dynamo.testing.CompileCounter()
# cfunc = torch.compile(func, backend=cnt, dynamic=True, fullgraph=True)
# assert cnt.frame_count == 0
# out = cfunc(torch.rand(12, 23))
# print(cnt.frame_count)


import torch
from torch._inductor.codecache import PyCodeCache

torch.compiler.reset()
PyCodeCache.clear()

def func(x):
    return x.sum()


cfunc = torch.compile(func)
out = cfunc(torch.rand(12, 23))

call_fn = None
found_call_fns = []
for key in PyCodeCache.cache:
    mod = PyCodeCache.cache[key]
    if "call" in mod.__dict__:
        found_call_fns.append((mod.__dict__["call"], mod.__dict__["__file__"]))

assert len(found_call_fns) == 1, (found_call_fns, f"{[(k, v.__file__) for k, v in PyCodeCache.cache.items()]}")
call_fn, fn_file = found_call_fns[0]
assert call_fn is not None
print(call_fn, fn_file)
