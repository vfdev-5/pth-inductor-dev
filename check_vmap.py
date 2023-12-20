import torch


def func(x):
    def fn(y):
        return torch.sum(y) + 1

    output = torch.vmap(fn)(x)
    return output


x = torch.arange(9).view(3, 3)

class LoggingTensor(torch.Tensor):
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # NOTE: Logging calls Tensor.__repr__, so we can't log __repr__ without infinite recursion
        if func is not torch.Tensor.__repr__:
            print(f"func: {func.__name__}, args: {args!r}, kwargs: {kwargs!r}")
        if kwargs is None:
            kwargs = {}
        return super().__torch_function__(func, types, args, kwargs)

# x = LoggingTensor(x)

print(func(x))

cfunc = torch.compile(func, fullgraph=True)
print(cfunc(x))
