# JAX

JAX on Aurora is experimental, and it is made available to the users through
the `frameworks` module

```bash
module load frameworks
python
>>> import jax
>>> jax.__version__
'0.4.30'
>>> 
>>> jax.devices()
INFO: Intel Extension for OpenXLA version: 0.5.0, commit: 70eb4b73
2025-05-07 05:00:03.591445: W xla/profiler/device_tracer_sycl.cc:281] ******************************Intel Extension For OpenXLA profiler Warning***************************************************
2025-05-07 05:00:03.591477: W xla/profiler/device_tracer_sycl.cc:285] Intel Extension For OpenXLA profiler not enabled, if you want to enable it, please set environment as :
export ZE_ENABLE_TRACING_LAYER=1
export UseCyclesPerSecondTimer=1

2025-05-07 05:00:03.591483: W xla/profiler/device_tracer_sycl.cc:290] ******************************************************************************************************
Platform 'sycl' is experimental and not all JAX functionality may be correctly supported!
[sycl(id=0), sycl(id=1), sycl(id=2), sycl(id=3), sycl(id=4), sycl(id=5), sycl(id=6), sycl(id=7), sycl(id=8), sycl(id=9), sycl(id=10), sycl(id=11)]
```
Currently, on Aurora JAX is functional at the single tile level. Multi-GPU,
Multi-node support is a work in progress.

# Examples

## Toy CNN
Below is a simple toy convolutional neural network example to get the users
started

```python
# Testing a toy network with conv+relu+layer-norm
from typing import Sequence
import time

import jax
import jax.numpy as jnp
from jax import random

import numpy as np

import flax.linen as nn


#from jax.config import config ## Deprecated on 0.4.25
jax.config.update("jax_enable_x64", True)

seed = 5
key1 = random.PRNGKey(seed)

INV_SCALE=1e6
REPEAT=10

# Small input
print("Running with small input size")
data = random.normal(key=key1, shape=(1, 2**12, 2**12, 1)) # (N, H, W, C) format
print(f"Size of the data in MB = {data.nbytes/1024/1024} \n")


# Medium input
#print("Running with Medium input size")
#data = random.normal(key=key1, shape=(1, 2**13, 2**13, 1)) # (N, H, W, C) format
#print(f"Size of the data in MB = {data.nbytes/1024/1024} \n")


# Large input
#print("Running with Large input size")
#data = random.normal(key=key1, shape=(1, 2**14, 2**14, 1)) # (N, H, W, C) format
#print(f"Size of the data in MB = {data.nbytes/1024/1024} \n")

data = jax.device_put(data)
print(data.dtype, data.devices())

class CNN(nn.Module):
  @nn.compact
  def __call__(self, x):
    x = nn.Conv(features=1, kernel_size=(3, 3), dtype=jnp.float64, param_dtype=jnp.float64)(x)
    #print(f"Shape after Conv layer 1 = {x.shape}")
    #print(f"Flattened x = {x.reshape((x.shape[0], -1))}")
    x = nn.relu(x)
    #print(f"Shape after relu 1 = {x.shape}")
    #print(f"Flattened x = {x.reshape((x.shape[0], -1))}")
    x = nn.LayerNorm(reduction_axes=(1, 2, 3), dtype=jnp.float64, param_dtype=jnp.float64)(x)
    #print(f"Shape after Layer Norm 1 = {x.shape}")
    #print(f"Flattened x = {x.reshape((x.shape[0], -1))}")
    return x

model = CNN()
variables = model.init(jax.random.key(0), data)
print(f"Variables = {variables}")
output = model.apply(variables, data).block_until_ready()

def repeat_jax(repeat):
    exec_times_jax_CNN = [0] * repeat
    for i in range(repeat):
        perf_count_start_jax_CNN = time.perf_counter_ns()
        output = model.apply(variables, data).block_until_ready()
        perf_count_end_jax_CNN = time.perf_counter_ns()
        exec_times_jax_CNN[i] = (perf_count_end_jax_CNN - perf_count_start_jax_CNN) / INV_SCALE
    return np.array([exec_times_jax_CNN,])

RESULTS=repeat_jax(repeat=REPEAT)

print(f"exec times JAX CNN  = {RESULTS[0]} in milliseconds \n")
```

## A JIT QKV-matrix multiplication
To run on Aurora and leverage the XPU devices we use the 
[Intel extension for OpenXLA](https://github.com/intel/intel-extension-for-openxla).
The following is an example provided from the above repository to demonstrate
the JIT functionality:
```python
# Copyright (c) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import jax
from jax import lax
from jax import random
import jax.numpy as jnp
import jax._src.test_util as jtu

import numpy as np

def seperateQKVGEMM(input, weight_q, weight_k, weight_v):
    out_q = jax.numpy.matmul(input, weight_q)
    out_k = jax.numpy.matmul(input, weight_k)
    out_v = jax.numpy.matmul(input, weight_v)
    return out_q, out_k, out_v

@jax.jit
def fusedQKVGEMM(input, weight_q, weight_k, weight_v):
    out_q = jax.numpy.matmul(input, weight_q)
    out_k = jax.numpy.matmul(input, weight_k)
    out_v = jax.numpy.matmul(input, weight_v)
    return out_q, out_k, out_v

def testQKVFusion():
    # Inputs
    m = 4
    k = 4096
    n = 4096
    key = jax.random.PRNGKey(1701)
    input = jax.random.uniform(key, (4, 4096)).astype(jnp.float16)
    weight_q = jax.random.uniform(key, (k, k)).astype(jnp.float16) # weights for Q
    print(f"Weight_q Device: {weight_q.devices()}")
    weight_k = jax.random.uniform(key, (k, k)).astype(jnp.float16) # weights for K
    print(f"Weight_k Device: {weight_k.devices()}")
    weight_v = jax.random.uniform(key, (k, k)).astype(jnp.float16) # weights for V
    print(f"Weight_v Device: {weight_v.devices()}")
    cpu_q, cpu_k, cpu_v = seperateQKVGEMM(input, weight_q, weight_k, weight_v)
    xpu_q, xpu_k, xpu_v = fusedQKVGEMM(input, weight_q, weight_k, weight_v)
    print(np.allclose(xpu_q, cpu_q, atol=1e-3, rtol=1e-3))
    print(np.allclose(xpu_k, cpu_k, atol=1e-3, rtol=1e-3))
    print(np.allclose(xpu_v, cpu_v, atol=1e-3, rtol=1e-3))

if __name__ == '__main__':
   testQKVFusion()
```
# Resources
- [Intel-extension-for-OpenXLA Docs](https://github.com/intel/intel-extension-for-openxla/tree/main/docs)
- [JAX Docs](https://docs.jax.dev/en/latest/)


