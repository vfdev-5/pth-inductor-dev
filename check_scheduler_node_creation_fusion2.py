#
# python -m debugpy --wait-for-client --listen 5678 check_scheduler_node_creation_fusion2.py
# TORCH_LOGS=+output_code python -u check_scheduler_node_creation_fusion2.py
#

from unittest.mock import patch
import torch
from torch._inductor.graph import GraphLowering
from torch._inductor import config


# Force multple scheduler nodes creation to fuse them
config.realize_opcount_threshold = 0


@torch.compile(fullgraph=True, dynamic=True)
def fn(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    o1 = x.permute((0, 1, 3, 2)) + y
    o2 = x.view(-1) - y.view(-1)
    return o1, o2


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


x = torch.rand(2, 3, 32, 32)
y = torch.rand(2, 3, 32, 32)

with patch.object(GraphLowering, "run_node", run_node_alt):
    # fn(x, w1, w2)
    fn(x, y)

# print("in_nodes:", in_nodes)
# print("outputs:", outputs)
print("graph_lowering_obj.buffers:", graph_lowering_obj.buffers)
print("graph_lowering_obj.scheduler:", graph_lowering_obj.scheduler.nodes)


# buffers = graph_lowering_obj.buffers
# scheduler = graph_lowering_obj.scheduler
# group_fn = scheduler.get_backend(torch.device("cpu")).group_fn
# snodes = [SchedulerNode(scheduler, n, group_fn) for n in buffers]
# print("snodes:", snodes)

