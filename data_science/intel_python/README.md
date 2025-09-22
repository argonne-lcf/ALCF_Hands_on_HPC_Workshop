# Intel's Data Parallel Extensions for Python (DPEP)

On Aurora, users can access Intel's Python stack comprising of compilers and libraries for programming heterogenous devices, namely the Data Parallel Extensions for Python (DPEP).
DPEP is composed of three main packages for programming on CPUs and GPUs:

- [dpnp](https://github.com/IntelPython/dpnp) - Data Parallel Extensions for Numpy is a library that implements a subset of Numpy that can be executed on any data parallel device. The subset is a drop-in replacement of core Numpy functions and numerical data types, similar to CuPy for CUDA devices.
- [dpctl](https://github.com/IntelPython/dpctl) - Data Parallel Control library provides utilities for device selection, allocation of data on devices, tensor data structure along with Python Array API Standard implementation, and support for creation of user-defined data-parallel extensions.
- [numba_dpex](https://github.com/IntelPython/numba-dpex) - Data Parallel Extensions for Numba is an extension to Numba compiler for programming data-parallel devices similar to developing programs with Numba for CPU or CUDA devices.


## Accessing the DPEP Packages

Users can access the dpnp, dpctl and numba-dpex packages by simply loading the latest AI/ML frameworks module with `module load frameworks`, thus enjoying interoperability with the ML and data science software stack.

```bash linenums="1"
module load frameworks
conda list | grep -E "dpnp|dpctl|numba"
```

???+ example "Output"

	``` { .bash .no-copy }
	dpctl                     0.20.2
    dpnp                      0.18.1
	numba                     0.60.0
    numba-dpex                0.23.0+0.g46e90f690.dirty
	```


## Compute-Follows-Data Programming Model

The DPEP packages follow the compute-follows-data programming model,
meaning that the offload target for a Python library call, or a hand-written kernel using numba-dpex,
does not need to be specified directly when making the call.
Instead, the offload target is inferred from the input arguments to the library call.
With this programming model, the user only needs to specify the offload target when creating the tensor/ndarray objects.

For example, visualize and execute the [01_compute_follow_data.py](./01_compute_follow_data.py) script.


## dpnp
The dpnp library implements the NumPy API using DPC++ and is meant to serve as a drop-in replacement for NumPy, similar to CuPy for CUDA devices.
Therefore, dpnp should be used to port NumPy and CuPy code to Intel GPU.
Dpnp offers good coverage of the NumPy and CuPy API, but please refer to this [comparison table](https://intelpython.github.io/dpnp/reference/comparison.html) to check the current status of dpnp parity.

All dpnp array creation routines and random number generators have additional optional keyword
arguments (device, queue, and usm_type) which users can leverage to explicitly specify on which device or queue
they want the data to be created along with the USM memory type to be used.

For example, visualize and execute the [02_dpnp_example.py](./02_dpnp_example.py) script.


## dpctl

The dpctl package lets users access devices supported by the DPC++ SYCL runtime.
The package exposes features such as device instrospection, execution queue creation, memory allocation, and kernel submission.
Below are some of the basic device management functions, but more functionality is available on the [dpctl documentation](https://intelpython.github.io/dpctl/latest/index.html).

For example, visualize and execute the [03_dpctl_introspection.py](./03_dpctl_introspection.py) script to get some information on the Aurora software stack and nodes.

The dpctl library also contains `dpctl.tensor`, which is a tensor library implemented using DPC++ that follows the Python Array API standard. 
It provides API for array creation, manipulation, and linear algebra functions, and is thus very similar to dpnp.
We refer the user to the [dpctl.tensor documentation](https://intelpython.github.io/dpctl/latest/api_reference/dpctl/tensor.html) for more details.


## Some Useful Notes on dpnp and dpctl Array Creation/Management

* Array creation API take as arguments the device, USM memory type and SYCL queue.
* The default device is GPU 0 with "device" memory type: `x = dpnp.asarray([1, 2, 3])`
* To allocate array on GPU 1: `x = dpnp.asarray([1, 2, 3], device=“gpu:1”)`
* To allocate array on USM: `x = dpnp.asarray([1, 2, 3], usm_type=“shared”)`
* dpnp and dpctl can be used to create a SYCL queue on a specific device and allocate an array on that `sycl::queue` object, such as
```python linenums="1"
import dpnp, dpctl

devices = dpctl.get_devices()
queue = dpctl.SyclQueue(devices[1])
arr = dp.ndarray([0,1,2],sycl_queue=queue)
```
or check out the [04_dpctl_queues.py](./04_dpctl_queues.py) script.
* After creating an array on the device, the array object carries the associated SYCL queue. When calling array manipulation/compute APIs, the output array is associated with the same `sycl::queue` and computations are scheduled for execution using this `sycl::queue`.
* Operating on arrays created on different devices or with different queues will raise an exception. Arrays can be copied across devices with `dpctl.tensor.asarray()`.


## Managing CPU and GPU Devices on Aurora

On Aurora, `ONEAPI_DEVICE_SELECTOR=level_zero:gpu` is set by default, meaning that the GPU are the only devices visible to dpctl and other applications using SYCL (can also confirm with `sycl-ls`).
For this reason, `dpctl.has_cpu_devices()` returns `False`.
This setting allows dpnp and dpctl to use the GPU as the default SYCL device without needing to explicitly specify it.
To access the CPU as a SYCL device, set `ONEAPI_DEVICE_SELECTOR=opencl:cpu`, or to make both CPU and GPU visible, set `ONEAPI_DEVICE_SELECTOR="opencl:cpu;level_zero:gpu"`. 

In addition, the number of GPU devices visible on each node depends on the `ZE_FLAT_DEVICE_HIERARCHY` environment variable, which is set to `FLAT` by the AI/ML frameworks module. 
With `ZE_FLAT_DEVICE_HIERARCHY=FLAT` 12 devices are visible (i.e., tile as device mode), whereas with `ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE` 6 devices are visible (GPU as device).
It is recommended to use `ZE_FLAT_DEVICE_HIERARCHY=FLAT`, especially if interoperability with PyTorch is needed, but `COMPOSITE` can be usefel if more memory is needed. 


## Timing dpnp and dpctl Kernels on GPU

Since `dpnp>0.15.0` and `dpctl>0.17.0`, all kernels are run asynchronously on the GPU with linear ordering of groups of tasks (similar to CuPy). 
This means that after the kernel is launched from the host to the device, the execution of the program does not wait for the kernel to finish and is instead allowed to proceed to the next lines of code. 
This model is desirable as it results in faster runtime and better GPU utilization.
However, it means that when timing the execution of dpnp and dpctl kernsl on the GPU, users must insert `.sycl_queue.wait()` before measuring the end time. 

For example, visualize and execute the [05_dpnp_timing.py](./05_dpnp_timing.py) script.


## numba-dpex

Numba-dpex is Intel's Data Parallel Extension for Numba which allows users to apply Numba's JIT compiler and generate performant, parallel code on Intel's GPU.
Its LLVM-based code generator implements a new kernel programming API (kapi) in pure Python that is modeled after the SYCL API.

The example [06_numba-dpex.py](./07_numba-dpex.py) implements and launches simple vector addition as a range kernel.
Range kernels implement a basic parallel-for calculation that is ideally suited for embarrassingly parallel operations, such as element-wise computations over n-dimensional arrays.


The `vecadd` function, when decorated as a dpex kernel, is compiled with numba-dpex into a data-parallel function to be executed individually by a set of work items (`#!python item.get_id(0)`).
Numba-dpex follows the SPMD programming model, wherein each work item runs the function for a subset of the elements of the input arrays.
The set of work items is defined by the `dpex.Range()` object and the `dpex.call_kernel()` call instructs every work item in the range to execute the `vecadd` kernel for a specific subset of the data.
Numba-dpex also follows the compute-follows-data programming model, meaning that the kernel is run on the same device as the dpnp and dpctl arrays passed as inputs.

Note that the numba-dpex kapi allows for more complex data parallel kernels (e.g., nd-range kernels) and the ability to create device callable functions.
For these and more features, we refer the users to the [numba-dpex documentation](https://intelpython.github.io/numba-dpex/latest/user_guide/kernel_programming/index.html#).


## DLPack

Thanks to dpctl supporting the Python Array API standard, both dpnp and dpctl provide interoperability with other Python libraries that follow the same standards, such as Numpy and PyTorch, through DLPack.
This allows for zero-copy data access across the Python ecosystem.

For example, visualize and execute the [07_dlpack.py](./07_dlpack.py) script which demonstreated interoperability between dpnp and PyTorch.

Notes about DLPack on Aurora:
* `ZE_FLAT_DEVICE_HIERARCHY` must be set to `FLAT`
* Zero-copy interoperability is supported between dpnp, dpctl, and PyTorch on CPU and GPU, and between Numpy as well on CPU only
* Interoperability between TensorFlow and the other packages is limited on the GPU due to TensorFlow not being compatible with the latest DLPack rules and still requiring the use of `dlcapsules`
* Numba-dpex does not directly support DLPack, however numba-dpex kernels take as inputs dpnp and dpctl arrays, thus inperoperability between PyTorch and numba-dpex is available through those packages
