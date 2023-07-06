# TORCH_COMPILE_DEBUG=1 python check_compile_tv_resize_v2.py
# TORCH_LOGS=+bytecode python check_compile_tv_resize_v2.py

import os
from typing import Optional, Tuple, Union

import torch
import torchvision

torchvision.disable_beta_transforms_warning()

if not ("OMP_NUM_THREADS" in os.environ):
    torch.set_num_threads(1)


print(torch.__version__, torchvision.__version__)


import numpy as np
from PIL import Image as PILImage

from torchvision.datapoints import Image, BoundingBox, Mask, BoundingBoxFormat
from torchvision.transforms.v2 import Resize



size = (64, 76)
# xyxy format
in_boxes = [
    [10, 15, 25, 35],
    [50, 5, 70, 22],
    [45, 46, 56, 62],
]
labels = [1, 2, 3]

im1 = 255 * np.ones(size + (3, ), dtype=np.uint8)
mask = np.zeros(size, dtype=np.int64)
for in_box, label in zip(in_boxes, labels):
    im1[in_box[1]:in_box[3], in_box[0]:in_box[2], :] = (127, max(127 * label, 255), max(127 * (label ** 2) - 50 * label, 255))
    mask[in_box[1]:in_box[3], in_box[0]:in_box[2]] = label

in_image = Image(torch.tensor(im1).permute(2, 0, 1).view(3, *size))
in_boxes = BoundingBox(in_boxes, format=BoundingBoxFormat.XYXY, spatial_size=size)
in_mask = Mask(torch.tensor(mask).view(1, *size))
in_image_tensor = 255 - in_image
in_image_pil = PILImage.fromarray(in_image_tensor.permute(1, 2, 0).cpu().numpy())
in_labels = labels

sample = (in_image, in_boxes, in_mask, in_image_tensor, in_image_pil, "abc", 123, in_labels)

transform = Resize(size=(34, 37), antialias=False)
c_transform = torch.compile(transform)

expected = transform(sample)
output = c_transform(sample)

out_image, out_boxes, out_mask, out_image_tensor, out_image_pil, out_text, out_number, out_labels = output
print(out_text, out_number, out_labels)

print(type(out_image))
print(type(out_boxes))
print(type(out_mask))

torch.testing.assert_close(expected[0], output[0])  # image
torch.testing.assert_close(expected[1], output[1])  # boxes
torch.testing.assert_close(expected[2], output[2])  # mask
torch.testing.assert_close(expected[3], output[3])  # image tensor