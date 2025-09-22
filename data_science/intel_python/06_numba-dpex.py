import dpnp
import numba_dpex as dpex
from numba_dpex import kernel_api as kapi

# Decorate the vecadd function as a dpex kernel
# dpex kernels implement a basic parallel-for calculation
# Each work item performs the same calculation on a subset of the data (SPMD model)
@dpex.kernel 
def vecadd(item: kapi.Item, a, b, c):
    i = item.get_id(0) # Get the work item
    c[i] = a[i] + b[i]

# Compute-follows-data programming model -- kernel executes on same device as input arrays
# Use dpnp and dpctl to create input/output arrays on desired device (default is GPU)
N = 1024 
a = dpnp.ones(N)
b = dpnp.ones_like(a)
c = dpnp.zeros_like(a)

# Define the number of work with dpex.Range(N)
# In this case, 1 item per array value 
work_items = dpex.Range(N)

# Call the kernel
# Every work item in the range executes the vecadd kernel for a subset of the data
dpex.call_kernel(vecadd, work_items, a, b, c)
assert dpnp.allclose(c,a+b)
print("Sum completed successfully")
