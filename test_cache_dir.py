import functools


a = [1, 2, 3, 4, 5]
a_iter = iter(a)

@functools.lru_cache(None)
def func():
    return next(a_iter)


for _ in range(5):
    func.cache_clear()
    print(func())
