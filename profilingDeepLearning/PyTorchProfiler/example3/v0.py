import torch
import torchvision.models as models

torch.set_default_tensor_type(torch.cuda.FloatTensor)
model = models.resnet18()
inputs = torch.randn(32, 3, 224, 224)

dir_name = './'

with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=0, warmup=1, active=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(dir_name)
        ) as prof:

    for _ in range(8):
        model(inputs)
        prof.step()
