# TORCH_LOGS=+dynamo TORCH_COMPILE_DEBUG=1 python -u check_smooth_l1_loss.py
# TORCH_COMPILE_DEBUG=1 python -m debugpy --wait-for-client --listen 5678 check_smooth_l1_loss.py

# TORCH_LOGS=+bytecode python -u check_smooth_l1_loss.py
# TORCH_LOGS=+graph,+graph_code python -u check_smooth_l1_loss.py
# TORCH_LOGS=+inductor python -u check_smooth_l1_loss.py

import logging

import torch
import torch._dynamo

torch._dynamo.config.verbose = True
# torch._dynamo.config.log_level = logging.DEBUG


def func(x, y, beta):

    loss = torch.nn.functional.smooth_l1_loss(x, y, beta=beta)
    return loss


def func0(x, y, beta):
    return torch.ops.aten.smooth_l1_loss(x, y, beta=beta)


def func2(x, y):

    loss = torch.nn.functional.l1_loss(x, y)
    return loss


def func3(x, y):

    loss = torch.nn.functional.mse_loss(x, y)
    return loss


device = "cuda"
x = torch.rand(4, 5, device=device)
y = torch.rand(4, 5, device=device)


# Compiled code: cat /tmp/torchinductor_root/nq/cnqtr7gzx6mqermp4clwhruiinnfcpnop5oumaex5ghl7ndy34af.py
#
# def call(args):
#     arg0_1, arg1_1 = args
#     args.clear()
#     with torch.cuda._DeviceGuard(0):
#         torch.cuda.set_device(0) # no-op to ensure context
#         buf0 = empty_strided((), (), device='cuda', dtype=torch.float32)
#         buf1 = buf0; del buf0  # reuse
#         stream0 = get_cuda_stream(0)
#         triton_per_fused_mse_loss_0.run(buf1, arg0_1, arg1_1, 1, 20, grid=grid(1), stream=stream0)
#         del arg0_1
#         del arg1_1
#         return (buf1, )

# cfunc3 = torch.compile(func3)
# cz3 = cfunc3(x, y)


# Compiled code: cat /tmp/torchinductor_root/gg/cggx4uzclik3v2mxws5v35llnljsstwbj6s4kxl3fx4ybdpeffpf.py
#
# def call(args):
#     arg0_1, arg1_1 = args
#     args.clear()
#     with torch.cuda._DeviceGuard(0):
#         torch.cuda.set_device(0) # no-op to ensure context
#         buf0 = empty_strided((), (), device='cuda', dtype=torch.float32)
#         buf1 = buf0; del buf0  # reuse
#         stream0 = get_cuda_stream(0)
#         triton_per_fused_abs_mean_sub_0.run(buf1, arg0_1, arg1_1, 1, 20, grid=grid(1), stream=stream0)
#         del arg0_1
#         del arg1_1
#         return (buf1, )


# cfunc2 = torch.compile(func2)
# cz2 = cfunc2(x, y)


# BEFORE THE PR:
# Compiled code: /tmp/torchinductor_root/gg/cggx4uzclik3v2mxws5v35llnljsstwbj6s4kxl3fx4ybdpeffpf.py
#
# def call(args):
#     arg0_1, arg1_1 = args
#     args.clear()
#     with torch.cuda._DeviceGuard(0):
#         torch.cuda.set_device(0) # no-op to ensure context
#         buf0 = aten.smooth_l1_loss(arg0_1, arg1_1)
#         del arg0_1
#         del arg1_1
#         buf1 = buf0
#         assert_size_stride(buf1, (), ())
#         del buf0
#         return (buf1, )
# IN THE PR:
#
# def call(args):
#     arg0_1, arg1_1 = args
#     args.clear()
#     with torch.cuda._DeviceGuard(0):
#         torch.cuda.set_device(0) # no-op to ensure context
#         buf0 = empty_strided((), (), device='cuda', dtype=torch.float32)
#         buf1 = buf0; del buf0  # reuse
#         stream0 = get_cuda_stream(0)
#         triton_per_fused_smooth_l1_loss_0.run(buf1, arg0_1, arg1_1, 1, 20, grid=grid(1), stream=stream0)
#         del arg0_1
#         del arg1_1
#         return (buf1, )

# cfunc = torch.compile(func)
# cz = cfunc(x, y, beta=0.0)


cfunc = torch.compile(func0)
cz = cfunc(x, y, beta=0.0)

# z = func(x, y)
# torch.testing.assert_close(z, cz)
