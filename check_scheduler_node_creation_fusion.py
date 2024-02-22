#
# python -m debugpy --wait-for-client --listen 5678 check_scheduler_node_creation_fusion.py
# TORCH_LOGS=+output_code python -u check_scheduler_node_creation_fusion.py
#

from unittest.mock import patch
import torch
from torch._inductor.graph import GraphLowering
from torch._inductor import config


# Force multple scheduler nodes creation to fuse them
config.realize_opcount_threshold = 0


# @torch.compile(fullgraph=True, dynamic=True)
# def fn(x: torch.Tensor, w1: torch.Tensor, w2: torch.Tensor) -> torch.Tensor:
#     o1 = x * w1.view(1, 1, 1, -1)
#     o2 = x * w2.view(1, 1, 1, -1)
#     output = o1 + o2
#     return output

@torch.compile(fullgraph=True, dynamic=True)
def fn(x: torch.Tensor, w1: torch.Tensor) -> torch.Tensor:
    o1 = x * w1.view(1, 1, 1, -1)
    output = o1 + 1.0
    return output


in_nodes = []
outputs = []
run_node = GraphLowering.run_node

graph_lowering_obj = None

def run_node_alt(self, n):
    global graph_lowering_obj

    graph_lowering_obj = self
    in_nodes.append(n)
    output = run_node(self, n)
    outputs.append(output)

    return output


x = torch.rand(2, 3, 32, 33)
w1 = torch.randn(33)
w2 = torch.randn(33)

with patch.object(GraphLowering, "run_node", run_node_alt):
    # fn(x, w1, w2)
    fn(x, w1)

# print("in_nodes:", in_nodes)
# print("outputs:", outputs)
print("graph_lowering_obj.buffers:", graph_lowering_obj.buffers)
print("graph_lowering_obj.scheduler:", graph_lowering_obj.scheduler.nodes)


# buffers = graph_lowering_obj.buffers
# scheduler = graph_lowering_obj.scheduler
# group_fn = scheduler.get_backend(torch.device("cpu")).group_fn
# snodes = [SchedulerNode(scheduler, n, group_fn) for n in buffers]
# print("snodes:", snodes)




"""
node
ComputedBuffer(name='buf0', layout=FixedLayout('cpu', torch.float32, size=[1, 3, 32, 32], stride=[3072, 1024, 32, 1]), data=Pointwise(
  'cpu',
  torch.float32,
  def inner_fn(index):
      _, i1, i2, i3 = index
      tmp0 = ops.load(arg1_1, i3 + 32 * i2 + 1024 * i1)
      tmp1 = ops.load(arg0_1, i3)
      tmp2 = tmp0 * tmp1
      return tmp2
  ,
  ranges=[1, 3, 32, 32],
  origin_node=mul,
  origins={mul}
))
body.reads_name2expr.keys()
dict_keys(['arg1_1', 'arg0_1'])
reordering_reindex
[<function same_reord...d19769040>, <function same_reord...d19769040>, <function same_reord...d19769040>]
x_vars, support_vars, sizes, reordering_reindex
([q0, q1, q2], [q0, q1, q2], [3, 32, 32], [<function same_reord...d19769040>, <function same_reord...d19769040>, <function same_reord...d19769040>])


Initial body before simplify_and_reorder
print(body.debug_str())

var_ranges = {q0: 3, q1: 32, q2: 32}
index0 = 1024*q0 + 32*q1 + q2
index1 = q2
def body(self, ops):
    get_index = self.get_index('index0')
    load = ops.load('arg1_1', get_index)
    get_index_1 = self.get_index('index1')
    load_1 = ops.load('arg2_1', get_index_1)
    mul = ops.mul(load, load_1)
    get_index_2 = self.get_index('index0')
    store = ops.store('buf1', get_index_2, mul, None)
    return store

->
"""


"""
if len(extra_indexing_symbols - indexing_symbols) > 0:
    extra_indexing_symbols = sorted(list(extra_indexing_symbols), key=lambda x: str(x))
    indexing_symbols = sorted(list(indexing_symbols), key=lambda x: str(x))

    symbols_map = {
        old: new for old, new in zip(reversed(extra_indexing_symbols), reversed(indexing_symbols))
    }
    extra_indexing_contraints = [
        c.subs(symbols_map) for c in extra_indexing_contraints
    ]
"""




##################################################################

# import torch
# from torch._inductor.ir import FixedLayout
# from torch._inductor.lowering import make_pointwise
# from torch._inductor.scheduler import SchedulerNode, ComputedBuffer


# size = []    # [1, s0, s3, s4]
# stride = []  # [s0*s3*s4, s3*s4, s4, 1])
# layout = FixedLayout("cpu", torch.float32, size=size, stride=stride)


# data = make_pointwise()


# node = ComputedBuffer(
#     name="test_buffer",
#     layout=layout,
#     data=data
# )




