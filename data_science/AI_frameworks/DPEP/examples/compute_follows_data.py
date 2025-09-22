import dpctl.tensor as dpt

x_gpu = dpt.arange(100, device="gpu") 
sqx_gpu = dpt.square(x_gpu) 
print(sqx_gpu.device) 

