# This example is adapted from
# https://pytorch.org/tutorials/beginner/profiler.html

import torch
import torch.autograd.profiler as profiler


class MyModule(torch.nn.Module):
    def __init__(
            self, in_features: int,
            out_features: int,
            hidden_sizes: list,
            bias: bool = True):
        super(MyModule, self).__init__()

        sizes = [in_features] + hidden_sizes + [out_features]
        layers = []
        for s in range(len(sizes)-1):
            layers.append(torch.nn.Linear(sizes[s], sizes[s+1], bias))
            layers.append(torch.nn.ReLU())
        self.linear = torch.nn.Sequential(*layers)

    def forward(self, input, mask):
        with profiler.record_function("LABEL1: linear pass"):
            out = self.linear(input)

        with profiler.record_function("LABEL2: masking"):
            threshold = out.sum(axis=1).mean()  # removed.item()
            hi_idx = (mask > threshold).nonzero(as_tuple=True)
        return out, hi_idx


torch.set_default_tensor_type(torch.cuda.FloatTensor)
model = MyModule(512, 8, [32, 32, 32])
input = torch.rand(512 * 512, 512)
mask = torch.rand((512, 512, 512))

# warm-up
model(input, mask)

with profiler.profile() as prof:
    out, idx = model(input, mask)
print(prof.key_averages().table(sort_by='self_cpu_time_total', row_limit=5))
