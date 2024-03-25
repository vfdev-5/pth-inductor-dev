
import torch
import torch.nn.functional as F
import hashlib

class DWModel(torch.nn.Module):
    def __init__(self):
        super(DWModel, self).__init__()

    def forward(self, hidden_states):
        return F.interpolate(hidden_states, scale_factor=2.0, mode="nearest")

torch.manual_seed(1234321)
model = DWModel()
model.to('cuda')
model = torch.compile(model, mode='max-autotune')

torch.manual_seed(1234321)
input = torch.randn((2, 640, 32, 32), generator=None, device='cuda', dtype=torch.float16, layout=torch.strided).to('cuda')

torch.manual_seed(1234321)
if True:
    res = model(input).clone()
    print(f"None          : {hashlib.sha256(res.cpu().numpy().tobytes()).hexdigest()}")

torch.manual_seed(1234321)
with torch.no_grad():
    res = model(input).clone()
    print(f"no_grad       : {hashlib.sha256(res.cpu().numpy().tobytes()).hexdigest()}")

torch.manual_seed(1234321)
with torch.inference_mode():
    res = model(input).clone()
    print(f"inference_mode: {hashlib.sha256(res.cpu().numpy().tobytes()).hexdigest()}")