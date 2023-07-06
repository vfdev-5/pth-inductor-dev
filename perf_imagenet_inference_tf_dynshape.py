from typing import Optional, Tuple, Union

import torch
import torchvision

torchvision.disable_beta_transforms_warning()


from torch import nn, Tensor
from torchvision.transforms.v2 import functional as F, InterpolationMode

import torch.utils.benchmark as benchmark


class ImageClassification(nn.Module):
    def __init__(
        self,
        *,
        crop_size: int,
        resize_size: int = 256,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        interpolation: InterpolationMode = InterpolationMode.BILINEAR,
        antialias: Optional[Union[str, bool]] = "warn",
    ) -> None:
        super().__init__()
        self.crop_size = [crop_size]
        self.resize_size = [resize_size]
        self.mean = list(mean)
        self.std = list(std)
        self.interpolation = interpolation
        self.antialias = antialias

    def forward(self, img: Tensor) -> Tensor:
        img = F.resize(img, self.resize_size, interpolation=self.interpolation, antialias=self.antialias)
        img = F.center_crop(img, self.crop_size)
        if not isinstance(img, Tensor):
            img = F.pil_to_tensor(img)
        img = F.convert_image_dtype(img, torch.float)
        img = F.normalize(img, mean=self.mean, std=self.std)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        format_string += f"\n    crop_size={self.crop_size}"
        format_string += f"\n    resize_size={self.resize_size}"
        format_string += f"\n    mean={self.mean}"
        format_string += f"\n    std={self.std}"
        format_string += f"\n    interpolation={self.interpolation}"
        format_string += "\n)"
        return format_string

    def describe(self) -> str:
        return (
            "Accepts ``PIL.Image``, batched ``(B, C, H, W)`` and single ``(C, H, W)`` image ``torch.Tensor`` objects. "
            f"The images are resized to ``resize_size={self.resize_size}`` using ``interpolation={self.interpolation}``, "
            f"followed by a central crop of ``crop_size={self.crop_size}``. Finally the values are first rescaled to "
            f"``[0.0, 1.0]`` and then normalized using ``mean={self.mean}`` and ``std={self.std}``."
        )


transform = ImageClassification(crop_size=224, antialias=False)
c_transform = torch.compile(transform, dynamic=True)


from time import perf_counter_ns


def benchmark_timer(
    fn, *, make_args, label, sub_label, description, min_run_time
):
    elapsed = 0.0
    times = []
    while elapsed < min_run_time * 1e9:

        args = make_args()

        start = perf_counter_ns()
        output = fn(*args)
        stop = perf_counter_ns()
        assert output is not None
        delta = stop - start
        elapsed += delta
        times.append(delta)

    times = torch.tensor(times).float() * 1e-3
    median = times.median().item()
    min, max = torch.aminmax(times)
    mean = torch.mean(times)

    print(
        label, sub_label, description, " : ", median, "µs (", min.item(), max.item(), ")", mean.item(),
    )


def main():

    results = []
    min_run_time = 10

    def make_args_fn(dtype, memory_format=torch.channels_last):
        def wrapper():
            h = 400 + torch.randint(-75, 75, size=()).item()
            w = 400 + torch.randint(-75, 75, size=()).item()
            x = torch.randint(0, 256, size=(1, 3, h, w), dtype=dtype)
            x = x.contiguous(memory_format=memory_format)[0]
            return (x, )
        return wrapper

    torch.manual_seed(12)
    torch.set_num_threads(1)
    for dtype in [torch.uint8, torch.float32]:

        x = torch.randint(0, 256, size=(1, 3, 345, 456), dtype=dtype)
        x = x.contiguous(memory_format=torch.channels_last)[0]

        output = c_transform(x)
        expected = transform(x)
        # torch.testing.assert_close(output, expected)

        benchmark_timer(
            transform,
            make_args=make_args_fn(dtype=dtype),
            label=f"Imagenet inference image transformation",
            sub_label=f"Input (3, randH, randW), {dtype}, CL",
            description="Eager",
            min_run_time=min_run_time
        )

        benchmark_timer(
            c_transform,
            make_args=make_args_fn(dtype=dtype),
            label=f"Imagenet inference image transformation",
            sub_label=f"Input (3, randH, randW), {dtype}, CL",
            description=f"Compiled",
            min_run_time=min_run_time
        )


if __name__ == "__main__":
    # import os
    # if not ("OMP_NUM_THREADS" in os.environ):
    #     torch.set_num_threads(1)

    print("")
    print(f"Torch version: {torch.__version__}")
    print(f"Torch config: {torch.__config__.show()}")
    # print(f"Num threads: {torch.get_num_threads()}")
    print(f"Torchvision version: {torchvision.__version__}")
    print("")

    main()