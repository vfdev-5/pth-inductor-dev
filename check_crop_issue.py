# TORCH_COMPILE_DEBUG=1 python -u check_crop_issue.py
import torch


def eager_v0(input, top, left, height, width):
    output = input - torch.tensor([left, top, left, top], dtype=input.dtype, device=input.device)
    output[..., 0::2].clamp_(min=0, max=width)
    output[..., 1::2].clamp_(min=0, max=height)
    return output


def eager_v1(input, top, left, height, width):
    output = input - torch.tensor([left, top, left, top], dtype=input.dtype, device=input.device)
    output[..., 0].clamp_(min=0, max=width)
    output[..., 2].clamp_(min=0, max=width)
    output[..., 1].clamp_(min=0, max=height)
    output[..., 3].clamp_(min=0, max=height)
    return output


def eager_v2(input, top, left, height, width):
    output = input - torch.tensor([left, top, left, top], dtype=input.dtype, device=input.device)
    output[..., [0, 2]].clamp_(min=0, max=width)
    output[..., [1, 3]].clamp_(min=0, max=height)
    return output


def eager_v3(input, top, left, height, width):
    left_top = torch.tensor([[left, top, left, top]], dtype=input.dtype, device=input.device)
    output = input - left_top
    output[..., 0::2].clamp_(min=0, max=width)
    output[..., 1::2].clamp_(min=0, max=height)
    return output


def fn_slicing_inplace(input, size):
    output = input.clone()
    output[..., 0::2].clamp_(0, size)
    output[..., 1::2].clamp_(0, size)
    return output


def fn_slicing_inplace2(x, y, s1, s2):
    output = x - y
    output[..., 0::2].clamp_(min=0, max=s1)
    output[..., 1::2].clamp_(min=0, max=s2)
    return output


def fn_slicing_inplace3(x, t, l, s1, s2):
    output = x - torch.tensor([[l, t, l, t]], dtype=x.dtype, device=x.device)
    output[..., 0::2].clamp_(min=0, max=s1)
    output[..., 1::2].clamp_(min=0, max=s2)
    return output


eager = eager_v3
kwargs = dict(top=7, left=3, height=3, width=5)
# input = torch.tensor([[0.0, 1.0, 10.0, 14.0]])
input = torch.randint(-5, 15, size=(1, 2, 3, 4))

# eager = fn_slicing_inplace
# input = torch.tensor([[-2.0, 1.0, 10.0, 14.0]])
# kwargs = dict(size=3)

# eager = fn_slicing_inplace2
# input = torch.tensor([[-2.0, 1.0, 10.0, 14.0]])
# y = torch.tensor([3, 7, 3, 7])
# kwargs = dict(y=y, s1=5, s2=3)

# eager = fn_slicing_inplace3
# # input = torch.tensor([[-2.0, 1.0, 10.0, 14.0]])
# input = torch.randint(-5, 15, size=(1, 2, 3, 4))
# kwargs = dict(t=7, l=3, s1=5, s2=3)

expected = eager(input, **kwargs)
# print("eager", expected)

torch._dynamo.reset()
compiled_static = torch.compile(eager, dynamic=False)
output_static = compiled_static(input, **kwargs)
# print("compiled static", output_static)


torch._dynamo.reset()
compiled_dynamic = torch.compile(eager, dynamic=True)
output_dynamic = compiled_static(input, **kwargs)
# print("compiled dynamic", output_dynamic)

torch.testing.assert_close(output_static, expected)
torch.testing.assert_close(output_static, output_dynamic)