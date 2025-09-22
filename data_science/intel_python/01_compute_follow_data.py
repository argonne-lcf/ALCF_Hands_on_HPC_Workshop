import dpctl.tensor as dpt

x_gpu = dpt.arange(100, device="gpu")
sqx_gpu = dpt.square(x_gpu) # square offloads to the “gpu” device
print(sqx_gpu.device) # sqx_gpu is created on the "gpu" device

# Expected output:
#	Device(level_zero:gpu:0)


