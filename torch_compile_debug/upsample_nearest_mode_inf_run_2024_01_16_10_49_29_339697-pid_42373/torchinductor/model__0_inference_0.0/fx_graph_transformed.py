class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[3, 640, 4, 4]"):
        # File: check_issue_upsample_nearest.py:12 in forward, code: return F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")
        upsample_nearest2d: "f32[3, 640, 8, 8]" = torch.ops.aten.upsample_nearest2d.default(arg0_1, [8, 8], 2.0, 2.0);  arg0_1 = None
        return (upsample_nearest2d,)
        