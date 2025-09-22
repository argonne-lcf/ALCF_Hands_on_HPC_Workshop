import dpnp as dp
import torch

t_ary = torch.arange(4).to('xpu') # array [0, 1, 2, 3] on GPU
dp_ary = dp.from_dlpack(t_ary)
t_ary[0] = -2.0 # modify the PyTorch array
print(f'Original PyTorch array: {t_ary}')
print(f'dpnp view of PyTorch array: {dp_ary} on device {dp_ary.device}\n')
del t_ary, dp_ary

dp_ary = dp.arange(4) # array [0, 1, 2, 3] on GPU
t_ary = torch.from_dlpack(dp_ary)
dp_ary[0] = -3.0 # modify the dpnp array
print(f'Original dpnp array: {dp_ary} on device {dp_ary.device}')
print(f'PyTorch view of dpnp array: {t_ary}')

# Expected output:
#   Original PyTorch array: tensor([-2,  1,  2,  3], device='xpu:0')
#   dpnp view of PyTorch array: [-2  1  2  3] on device Device(level_zero:gpu:0)
#
#   Original dpnp array: [-3  1  2  3] on device Device(level_zero:gpu:0)
#   PyTorch view of dpnp array: tensor([-3,  1,  2,  3], device='xpu:0')
