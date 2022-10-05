# This example is adapted from
# https://pytorch.org/tutorials/beginner/profiler.html

import torch
import numpy as np
import cProfile


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
        out = self.linear(input)

        threshold = out.sum(axis=1).mean().item()
        hi_idx = np.argwhere(mask.cpu().numpy() > threshold)
        hi_idx = torch.from_numpy(hi_idx).cuda()

        return out, hi_idx


torch.set_default_tensor_type(torch.cuda.FloatTensor)
model = MyModule(512, 8, [32, 32, 32])
input = torch.rand(512 * 512, 512)
mask = torch.rand((512, 512, 512))

# warm-up
model(input, mask)

with cProfile.Profile() as pr:
    out, idx = model(input, mask)

pr.print_stats('cumulative')
