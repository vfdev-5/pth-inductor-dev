import torch
from torch._inductor import config
config.debug = True

device = "cpu"
# device = "cuda"

print("device:", device)


def indexit(A, b):
    return A[b]


c_indexit = torch.compile(indexit, backend="inductor")

A = torch.arange(20, device=device)

print("\n--- Indirect indexing")
b = torch.tensor([-1, -1], device=device)
print("non compiled:", indexit(A, b))

try:
    print("compiled:", c_indexit(A, b))
except RuntimeError as e:
    print("Raised runtime error:", e)


b = torch.tensor([20, 20], device=device)
try:
    print("non compiled:", indexit(A, b))
except IndexError as e:
    print("Non-compiled fn raised IndexError:", e)

try:
    print("compiled:", c_indexit(A, b))
except IndexError as e:
    print("Compiled fn raised IndexError:", e)


print("\n--- Direct indexing")

try:
    print("compiled:", c_indexit(A, -1))
except IndexError as e:
    print("Compiled fn raised IndexError:", e)

try:
    print("compiled:", c_indexit(A, 20))
except IndexError as e:
    print("Compiled fn raised IndexError:", e)
