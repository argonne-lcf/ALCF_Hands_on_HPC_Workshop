# Description

The goal of this implementation is to provide an example of how one can integrate a python-based, machine learning (ML) framework within a computational physics (PDE) solver.  Like most GPU-enabled solvers, the physics kernel is executed on the device where critical field data resides. This implementation makes use of the [CuPY](https://cupy.dev/) framework to perform in-situ analysis on the device, thereby, avoiding the cost of data movement to host. Furthermore, this example demonstrates how to couple the ML workflow with an application that uses a performance-portability abstraction layer, namely [OCCA](https://github.com/libocca/occa), which executes physics kernels on the device for a variety backend-specific programming models (e.g. CUDA, HIP, SYCL).    

## Requirements

- [OCCA](https://github.com/libocca/occa)
- cmake
- C++17 compiler
- C11 compiler
- CUDA 9 or later
- Virtual Python Environment

All of the above are provided on ThetaGPU

## Building and Running 

We assume that you have cloned the repo to a suitable location. These are the steps to execute this code on ThetaGPU (interactively):

1. From the theta login node, please Login to a ThetaGPU service node
```
ssh thetagpusn1
```
2. Request an interactive session on an A100 GPU
```
qsub -A SDL_Workshop \
     -q training-gpu \
     -I \
     -n 1 \
     --attrs filesystems=home,grand,eagle \
     -t 60
```
3. Set the Environment

You can do `source set_OCCA_env.sh`. This loads modules and sets certain environment variables. 

```
$ cat set_OCCA_env.sh

module load cmake
module load conda/2021-11-30

SDL=sdl_workshop/05_Simulation_ML/ML_PythonC++_Embedding/ThetaGPU_OCCA/

export OCCA_DIR=/lus/grand/projects/catalyst/world-shared/spatel/occa/install
export OCCA_CACHE_DIR=/path/to/$SDL

export OCCA_CXX="g++"
export OCCA_CXXFLAGS="-O3"
export OCCA_CUDA_COMPILER="nvcc"
export OCCA_CUDA_COMPILER_FLAGS="-O3 --fmad=true"

export PATH+=":${OCCA_DIR}/bin"
export LD_LIBRARY_PATH+=":${OCCA_DIR}/lib"
```
NOTE: the `OCCA_CACHE_DIR` specifies a location where OCCA-specific kernels (.okl) are stored/cached. Please make note of your path to this directory. 

4. Check the environment.
```
$ module list

Currently Loaded Modules:
  1) Core/StdEnv   2) cmake/3.19.5   3) openmpi/openmpi-4.1.4_ucx-1.12.1_gcc-9.4.0   4) conda/2021-11-30

```
```
$ occa info
    ========+======================+=================================
     CPU(s) | Processor Name       | AMD EPYC 7742 64-Core Processor 
            | Memory               | 1007.69 GB                      
            | Clock Frequency      | 2.2 MHz                         
            | SIMD Instruction Set | SSE2                            
            | SIMD Width           | 128 bits                        
            | L1d Cache Size       |   4 MB                          
            | L1i Cache Size       |   4 MB                          
            | L2 Cache Size        |  64 MB                          
            | L3 Cache Size        | 512 MB                          
    ========+======================+=================================
     CUDA   | Device Name          | NVIDIA A100-SXM4-40GB           
            | Device ID            | 0                               
            | Memory               | 39.58 GB                        
    ========+======================+=================================
```
5. Activate the virtual Python environment

```
$ conda activate
```
6. Build by running the `build.sh` script from this directory. This script is the cmake driver.  

```
$ sh build.sh
```
You should some output that looks something like:
```
-- Build files have been written to: ... 
+ cmake --build .../ML_PythonC++_Embedding/ThetaGPU_OCCA/build --parallel 4
Scanning dependencies of target burger
[ 50%] Building CXX object CMakeFiles/burger.dir/main.cpp.o
[100%] Linking CXX executable burger
[100%] Built target burger
+ cmake --install ... --prefix ... 
-- Install configuration: "RelWithDebInfo"
-- Installing: .../ML_PythonC++_Embedding/ThetaGPU_OCCA/install/./burger
-- Set runtime path of ".../ThetaGPU_OCCA/install/./burger" to ""
-- Installing: .../ML_PythonC++_Embedding/ThetaGPU_OCCA/install/kernel
-- Installing: .../ML_PythonC++_Embedding/ThetaGPU_OCCA/install/kernel/burger.okl
-- Installing: .../ML_PythonC++_Embedding/ThetaGPU_OCCA/install/./python_module.py
```
7. Run

```
$ cd install/
$ ls
burger	kernel	python_module.py

$ ls kernel/
burger.okl

$ ./burger
Initialization of Python: Done
Within Python Module File
Loaded Python Module File: Done
Loaded Functions: Done
Called python data collection function successfully
time = 0.001
Called python data collection function successfully
Called python data collection function successfully
Called python data collection function successfully
...
...
...
Called python data collection function successfully
Called python data collection function successfully
time = 2.001
Mean Wall-Time: 0.0469978
A random value in the solution array: 0.0733873
Called python analyses function successfully
Performing SVD
Called python analyses function successfully

$ ls
burger	Field_evolution.png  kernel python_module.py  SVD_Eigenvectors.png
```

## Key Features

### OCCA API: Host, Device & Memory
The host, usually a CPU processor, is the physical device that runs the application. A device can be a physical device which can be the host (i.e. a CPU) or an offload device - one that is physically distinct from the host.  In this example, our kernels run on an offload device. OCCA enables the user to connect to the physical device through the OCCA API.  For example, the following snippet is one way to instantiate the device:
```
  occa::device device;

  device.setup((std::string) args["options/device"]);
  device.setup({
     {"mode"     , "CUDA"},
     {"device_id", 0},
   });   
```
The OCCA API is also used for memory allocation.  This is done using the [malloc](https://libocca.org/#/api/device/malloc) method on a device object. 
```
  int N = 1000;
  occa::memory o_uh, o_uh_prev;

  o_uh = device.malloc<double>(N);
  o_uh_prev = device.malloc<double>(N);

```
To get the backend pointer, one can do:
```
  double *d_b = static_cast<double *>(o_uh.ptr());
```
This exposes the address set by the backend model (e.g. CUDA) and hardware (e.g. NVIDIA A100). Doing this will be helpful as we perform in-situ analysis on data resident on the device.  

### Using CuPY to enable zero-copy, in-situ analysis
In CuPY, `cupy.ndarray` is the counterpart of the NumPy `numpy.ndarray` which provides an interface for fixed-size multi-dimensional array which resides on a CUDA device.  Low-level CUDA support in CuPY allows us to retreive device memory. For example,

```
import cupy
from cupy.cuda import memory

  def my_function(a):
      b = cupy.ndarray(
                  a.__array_interface__['shape'][0],
                  cupy.dtype(a.dtype.name),
                  cupy.cuda.MemoryPointer(cupy.cuda.UnownedMemory(
                                             a.__array_interface__['data'][0], #<---Pointer?
                                             a.size,
                                             a,
                                             0), 0),
                  strides=a.__array_interface__['strides'])

```
