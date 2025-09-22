import dpnp as np
from time import perf_counter

x = np.random.randn(1000,1000)
tic = perf_counter()
y = np.matmul(x,x)
y.sycl_queue.wait()
print(f"Execution time: {perf_counter() - tic} sec")

