import torch
from torch._functorch.compile_utils import fx_graph_cse


def func(inpt, osize):
    size = inpt.shape[-1]
    s1 = size - 1
    s2 = size - 1.0
    scale = s2 / (osize - 1.0)
    inpt = torch.clamp(inpt, 0, s1)
    return scale * inpt

gms = []
def toy_backend(gm, _):
    gms.append(gm)
    return gm.forward

# torch._dynamo.reset()
# fn = torch.compile(backend=toy_backend, dynamic=True)(func)
# t = torch.rand(3, 100)
# out = fn(t, 50)
# gm = gms[0]

from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.backends.common import aot_autograd


toy_aot_backend = aot_autograd(
    fw_compiler=toy_backend, partition_fn=min_cut_rematerialization_partition
)


print(gm.graph)
new_fx_g = fx_graph_cse(gm.graph)
print(str(new_fx_g))





exit(0)
#########
aten = torch._ops.ops.aten

def func(input, osize, true_ac):
    dtype = torch.float32
    i = torch.arange(osize, device=input.device).to(dtype=dtype)

    in_size = input.shape[-1]
    if true_ac:
        scale = (in_size - 1.0) / (osize - 1.0)
        x_f32 = scale * i
    else:
        scale = (in_size - 1) / (osize - 1)
        x_f32 = scale * i

    x = x_f32.floor().to(torch.int64)
    x = torch.clamp(x, 0, in_size - 1)

    output = 0.5 * aten._unsafe_index(input, [None, x])
    return output



# backend = "eager"
# backend = "aot_eager_decomp_partition"
backend = "aot_eager"
# backend = "inductor"

c_func = torch.compile(func, backend=backend, dynamic=True, fullgraph=True)

t = torch.rand(3, 100, requires_grad=True)
# t = torch.rand(3, 100)

# expected = func(t, 50, True)
# output1 = c_func(t, 50, False)
output2 = c_func(t, 50, True)
# torch.testing.assert_close(expected, output1)
# torch.testing.assert_close(output1, output2)


# ## No needed anymore for repro

# from torch.testing._internal.optests import aot_autograd_check

# aot_autograd_check(
#     func,
#     (t, 50, False),
#     {},
#     dynamic=True,
#     check_gradients=True,
#     try_check_data_specialization=False
# )


# aot_autograd_check(
#     func,
#     (t, 50, True),
#     {},
#     dynamic=True,
#     check_gradients=False,
#     try_check_data_specialization=False
# )



####### Notes
"""
fw_module from pytorch/torch/_functorch/_aot_autograd/jit_compile_runtime_wrappers.py::aot_dispatch_autograd

```
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "Sym(s0)", primals_2: "Sym(s1)", primals_3: "f32[s0, s1]", primals_4: "Sym(s2)"):
        # File: check_clamp_issue.py:8 in func, code: i = torch.arange(osize, device=input.device).to(dtype=dtype)
        arange: "i64[s2]" = torch.ops.aten.arange.default(primals_4, device = device(type='cpu'), pin_memory = False)
        _to_copy: "f32[s2]" = torch.ops.aten._to_copy.default(arange, dtype = torch.float32);  arange = None

        # File: check_clamp_issue.py:13 in func, code: x_f32 = scale * i
        sub: "Sym(s1 - 1.0)" = primals_2 - 1.0
        sub_1: "Sym(s2 - 1.0)" = primals_4 - 1.0
        truediv: "Sym(s1/(s2 - 1.0) - 1.0/(s2 - 1.0))" = sub / sub_1;  sub_1 = None
        mul: "f32[s2]" = torch.ops.aten.mul.Tensor(_to_copy, truediv);  _to_copy = truediv = None

        # File: check_clamp_issue.py:18 in func, code: x = x_f32.floor().to(torch.int64)
        floor: "f32[s2]" = torch.ops.aten.floor.default(mul);  mul = None
        _to_copy_1: "i64[s2]" = torch.ops.aten._to_copy.default(floor, dtype = torch.int64);  floor = None

        # File: check_clamp_issue.py:19 in func, code: x = torch.clamp(x, 0, in_size - 1)
        clamp: "i64[s2]" = torch.ops.aten.clamp.default(_to_copy_1, 0, sub);  _to_copy_1 = sub = None

        # File: check_clamp_issue.py:21 in func, code: output = 0.5 * aten._unsafe_index(input, [None, x])
        _unsafe_index: "f32[s0, s2]" = torch.ops.aten._unsafe_index.Tensor(primals_3, [None, clamp]);  primals_3 = None
        mul_1: "f32[s0, s2]" = torch.ops.aten.mul.Tensor(_unsafe_index, 0.5);  _unsafe_index = None
        return [mul_1, clamp, primals_1, primals_2, primals_4]
```


node = clamp
args = (tensor_int64, 0, 99.0)
Why 99.0 is float ? -> because sub_1: "Sym(s2 - 1.0)" = primals_4 - 1.0

node.target is <OpOverload(op='aten.clamp', overload='default')>

run_node(node) -> output dtype is float32 ???


Failure is related to functorch CSE config and more exactly:
```
nn.args
(primals_2, 1.0)
n.args
(primals_2, 1)
hash(n.args) == hash(nn.args)
True
```
"""