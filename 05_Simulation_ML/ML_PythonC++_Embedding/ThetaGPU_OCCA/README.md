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

Load the cmake and virtual python environment (THIS MIGHT HAVE TO CHANGE) 
```
module load cmake
module load conda/2021-11-30
```
Set the proper paths for the OCCA library (THIS MIGHT HAVE TO CHANGE)
```
export OCCA_DIR=/lus/eagle/projects/catalyst/spatel/work/test_occa/occa/install/
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
```
```
$ occa env
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


```


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

