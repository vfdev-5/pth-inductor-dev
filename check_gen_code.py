import torch

##########################################################################################

# # buf0: SchedulerNode(ComputedBuffer)
# # buf0.writes = [MemoryDep('buf0', 0, {})]
# # buf0.unmet_dependencies = []
# # buf0.met_dependencies = []
# # buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
# # buf0.group.device = cpu
# # buf0.group.iteration = ((), (s0,))
# # buf0.sizes = ([], [s0])
# # class buf0_loop_body:
# #     var_ranges = {z0: s0}
# #     index0 = z0 + 5
# #     index1 = 0
# #     def body(self, ops):
# #         get_index = self.get_index('index0')
# #         index_expr = ops.index_expr(get_index, torch.int64)
# #         reduction = ops.reduction(torch.int64, torch.int64, 'sum', index_expr)
# #         get_index_1 = self.get_index('index1')
# #         store_reduction = ops.store_reduction('buf0', get_index_1, reduction)
# #         return store_reduction

# def func(n):
#     x = torch.arange(n)
#     x = x + 5
#     return x.sum()

# c_func = torch.compile(func)
# c_func(10)
# c_func(20)


##########################################################################################


def func(n, m):
    x = torch.arange(n)
    x = x + 5

    y = torch.arange(m)
    y = y - 6
    a = x.sum()
    b = y.sum()
    return x * y

c_func = torch.compile(func)
c_func(10, 11)
c_func(20, 21)
