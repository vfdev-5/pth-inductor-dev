import math
import torch


def f3_a(a: float) -> float:
    return math.floor(a)


def f3_b(a: float) -> float:
    a = torch.tensor(a)
    return torch.floor(a).item()


def f2(a: float) -> float:
    if torch._dynamo.is_compiling():
        return f3_a(a) * 10.0
    else:
        return f3_b(a) * 0.1


sf1 = torch.jit.script(f2)

# cf1 = torch.compile(f2)

# o1 = cf1(1.0)
# o2 = sf1(1.0)

# print(o1, o2)
