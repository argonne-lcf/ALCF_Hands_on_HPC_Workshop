import dpnp 
import numba_dpex as dpex 
from numba_dpex import kernel_api as kapi 

# Data parallel kernel implementation of vector sum
@dpex.kernel
def vecadd(item: kapi.Item, a, b, c):
    i = item.get_id(0)
    c[i] = a[i] + b[i]

N = 1024
a = dpnp.ones(N) 
b = dpnp.ones_like(a) 
c = dpnp.zeros_like(a) 
dpex.call_kernel(vecadd, dpex.Range(N), a, b, c)
