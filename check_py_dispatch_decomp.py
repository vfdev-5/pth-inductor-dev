
import torch


def fn(x):
    out = torch.nn.functional.interpolate(x, (5, 5), mode="bicubic")
    return out


def check(req_grad):
    print("-- Call fn on cpu tensor")
    x = torch.arange(4 * 3 * 10 * 10, dtype=torch.float32).reshape(4, 3, 10, 10)
    if req_grad:
        x = x.requires_grad_()
    expected = fn(x)
    print(expected.shape, expected.dtype, expected.device)

    print("-- Call fn on meta tensor")
    x = torch.arange(4 * 3 * 10 * 10, dtype=torch.float32, device="meta").reshape(4, 3, 10, 10)
    if req_grad:
        x = x.requires_grad_()

    expected = fn(x)
    print(expected.shape, expected.dtype, expected.device)

    print("-- Call compiled fn on cpu tensor")
    x = torch.arange(4 * 3 * 10 * 10, dtype=torch.float32).reshape(4, 3, 10, 10)
    if req_grad:
        x = x.requires_grad_()

    cfn = torch.compile(fn)
    output = cfn(x)
    print(output.shape, output.dtype, output.device)


print("\n---- Grad mode, input without grad")
check(req_grad=False)

print("\n---- Grad mode, input with grad")
check(req_grad=True)

with torch.no_grad():
    print("\n---- NoGrad mode, input without grad")
    check(req_grad=False)

    print("\n---- NoGrad mode, input with grad")
    check(req_grad=True)

with torch.inference_mode():
    print("\n---- Inf mode, input without grad")
    check(req_grad=False)

    print("\n---- Inf mode, input with grad")
    check(req_grad=True)


# Output:
"""
---- Grad mode, input without grad
-- Call fn on cpu tensor
Call C++ upsample_nearest1d_kernel_impl
torch.Size([4, 3, 5]) torch.float32 cpu
-- Call fn on meta tensor
Call meta upsample_nearest1d
torch.Size([4, 3, 5]) torch.float32 meta
-- Call compiled fn on cpu tensor
call decomp upsample_nearest1d_vec <class 'torch._subclasses.fake_tensor.FakeTensor'> <class 'list'>
call decomp upsample_nearest1d:  <class 'torch._subclasses.fake_tensor.FakeTensor'> <class 'list'>
call decomp upsample_nearest1d_vec <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
call decomp upsample_nearest1d:  <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
call decomp upsample_nearest1d_vec <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
call decomp upsample_nearest1d:  <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
torch.Size([4, 3, 5]) torch.float32 cpu

---- Grad mode, input with grad
-- Call fn on cpu tensor
Call C++ upsample_nearest1d_kernel_impl
torch.Size([4, 3, 5]) torch.float32 cpu
-- Call fn on meta tensor
Call meta upsample_nearest1d
torch.Size([4, 3, 5]) torch.float32 meta
-- Call compiled fn on cpu tensor
call decomp upsample_nearest1d_vec <class 'torch._subclasses.fake_tensor.FakeTensor'> <class 'list'>
call decomp upsample_nearest1d:  <class 'torch._subclasses.fake_tensor.FakeTensor'> <class 'list'>
call decomp upsample_nearest1d_vec <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
call decomp upsample_nearest1d:  <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
call decomp upsample_nearest1d_vec <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
call decomp upsample_nearest1d:  <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
torch.Size([4, 3, 5]) torch.float32 cpu

---- NoGrad mode, input without grad
-- Call fn on cpu tensor
Call C++ upsample_nearest1d_kernel_impl
torch.Size([4, 3, 5]) torch.float32 cpu
-- Call fn on meta tensor
Call meta upsample_nearest1d
torch.Size([4, 3, 5]) torch.float32 meta
-- Call compiled fn on cpu tensor
call decomp upsample_nearest1d_vec <class 'torch._subclasses.fake_tensor.FakeTensor'> <class 'list'>
call decomp upsample_nearest1d:  <class 'torch._subclasses.fake_tensor.FakeTensor'> <class 'list'>
call decomp upsample_nearest1d_vec <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
call decomp upsample_nearest1d:  <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
call decomp upsample_nearest1d_vec <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
call decomp upsample_nearest1d:  <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
torch.Size([4, 3, 5]) torch.float32 cpu

---- NoGrad mode, input with grad
-- Call fn on cpu tensor
Call C++ upsample_nearest1d_kernel_impl
torch.Size([4, 3, 5]) torch.float32 cpu
-- Call fn on meta tensor
Call meta upsample_nearest1d
torch.Size([4, 3, 5]) torch.float32 meta
-- Call compiled fn on cpu tensor
call decomp upsample_nearest1d_vec <class 'torch._subclasses.fake_tensor.FakeTensor'> <class 'list'>
call decomp upsample_nearest1d:  <class 'torch._subclasses.fake_tensor.FakeTensor'> <class 'list'>
call decomp upsample_nearest1d_vec <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
call decomp upsample_nearest1d:  <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
call decomp upsample_nearest1d_vec <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
call decomp upsample_nearest1d:  <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
torch.Size([4, 3, 5]) torch.float32 cpu

---- Inf mode, input without grad
-- Call fn on cpu tensor
Call C++ upsample_nearest1d_kernel_impl
torch.Size([4, 3, 5]) torch.float32 cpu
-- Call fn on meta tensor
Call meta upsample_nearest1d
torch.Size([4, 3, 5]) torch.float32 meta
-- Call compiled fn on cpu tensor
call decomp upsample_nearest1d_vec <class 'torch._subclasses.fake_tensor.FakeTensor'> <class 'list'>
Call meta upsample_nearest1d
call decomp upsample_nearest1d_vec <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
call decomp upsample_nearest1d:  <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
call decomp upsample_nearest1d_vec <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
call decomp upsample_nearest1d:  <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
torch.Size([4, 3, 5]) torch.float32 cpu

---- Inf mode, input with grad
-- Call fn on cpu tensor
Call C++ upsample_nearest1d_kernel_impl
torch.Size([4, 3, 5]) torch.float32 cpu
-- Call fn on meta tensor
Call meta upsample_nearest1d
torch.Size([4, 3, 5]) torch.float32 meta
-- Call compiled fn on cpu tensor
call decomp upsample_nearest1d_vec <class 'torch._subclasses.fake_tensor.FakeTensor'> <class 'list'>
Call meta upsample_nearest1d
call decomp upsample_nearest1d_vec <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
call decomp upsample_nearest1d:  <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
call decomp upsample_nearest1d_vec <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
call decomp upsample_nearest1d:  <class 'torch._subclasses.functional_tensor.FunctionalTensor'> <class 'list'>
torch.Size([4, 3, 5]) torch.float32 cpu
"""