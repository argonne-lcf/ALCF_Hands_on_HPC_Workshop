# Description

The goal of this implementation is to provide an example of how one can integrate a python-based, machine learning (ML) framework within a computational physics (PDE) solver.  Like most GPU-enabled solvers, the physics kernel is executed on the device where critical field data resides. This implementation makes use of the [CuPY](https://cupy.dev/) framework to perform in-situ analysis on the device, thereby, avoiding the cost of data movement to host. Furthermore, this example demonstrates how to couple the ML workflow with an application that uses a performance-portability abstraction layer, namely [OCCA](https://github.com/libocca/occa), which executes physics kernels on the device for a variety backend-specific programming models (e.g. CUDA, HIP, SYCL).    

## Requirements

- [OCCA](https://github.com/libocca/occa)
- C++17 compiler
- C11 compiler
- CUDA 9 or later
- Virtual Python Environment

All of the above are provided on ThetaGPU

## Building and Running 

We assume that you have cloned the repo to a suitable location. These are the steps to execute this code on ThetaGPU (interactively):
1. Login to a ThetaGPU head node
```
ssh thetagpusn1
```
2. Request an interactive session on an A100 GPU
```
qsub -n 1 -q training-gpu -A SDL_Workshop -I -t 1:00:00
```
3. Set the Environment

You can do `source set_OCCA_env.sh`. Which load modules and sets certain environment variables. 

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

Activate the virtual Python environment
```
conda activate
```
Check the environment. (UPDATE THIS SECTION)
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

4. Compile



5. Run


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



## Acknowledgements


We assume that you have cloned the repo to a suitable location. These are the steps to execute this code on ThetaGPU (interactively):
1. Login to a ThetaGPU head node
```
ssh thetagpusn1
```
2. Request an interactive session on an A100 GPU
```
qsub -n 1 -q training-gpu -A SDL_Workshop -I -t 1:00:00
```
Following this, we need to execute a few commands to get setup with an appropriately optimized tensorflow. These are:

3. Activate the TensorFlow 2.2 singularity container:
```
singularity exec -B /lus:/lus --nv /lus/theta-fs0/software/thetagpu/nvidia-containers/tensorflow2/tf2_20.08-py3.simg bash
```
4. Setup access to the internet
```
export http_proxy=http://theta-proxy.tmi.alcf.anl.gov:3128
export https_proxy=https://theta-proxy.tmi.alcf.anl.gov:3128
```
Now that we can access the internet, we need to set up a virtual environment in Python (these commands should only be run the first time)
```
python -m pip install --user virtualenv
export VENV_LOCATION=/home/$USER/THETAGPU_TF_ENV # Add your path here
python -m virtualenv --system-site-packages $VENV_LOCATION
source $VENV_LOCATION/bin/activate
python -m pip install cmake
python -m pip install matplotlib
python -m pip install sklearn
```
If you experience SSL errors during the CMake install, you can set `export https_proxy=http://theta-proxy.tmi.alcf.anl.gov:3128` (http instead of https on the URL) and retry.

In the future, you only need to reactivate the environment:
```
source $VENV_LOCATION/bin/activate
```
5. Now we are ready to build our executable by executing the provided shell script (within the cloned repo):
```
source setup.sh
```
6. You can now run the app using
```
mpirun -n 1 -npernode 1 -hostfile $COBALT_NODEFILE ./app
```

### To run on the queue (and not interactively)
```
qsub submit.sh
```
Note that you need to inspect the `queue_submission.sh` script to make sure it is _your_ virtual environment that is used. The submission script being used here assumes that you have set up a virtual environment with cmake, matplotlib, sklearn, etc. (i.e., steps 1,2,3,4 should have been executed previously)

