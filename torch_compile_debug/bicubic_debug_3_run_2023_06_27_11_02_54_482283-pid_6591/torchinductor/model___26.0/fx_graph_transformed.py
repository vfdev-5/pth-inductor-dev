class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: f32[1, 3, 32, 32]):
        # File: check_interpolate_bicubic.py:14, code: img = torch.nn.functional.interpolate(img, size=(12, 32), mode="bicubic", antialias=False)
        upsample_bicubic2d: f32[1, 3, 12, 32] = torch.ops.aten.upsample_bicubic2d.default(arg0_1, [12, 32], False);  arg0_1 = None
        return (upsample_bicubic2d,)
        