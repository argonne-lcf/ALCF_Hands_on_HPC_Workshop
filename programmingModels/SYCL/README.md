# SYCL on ALCF Polaris & Aurora – Quickstart Guide

This guide provides a practical introduction to compiling and running **SYCL** applications on **ALCF systems**:

- **Polaris** → NVIDIA A100 GPUs (via SYCL's oneAPI CUDA backend)
- **Aurora** → Intel Ponte Vecchio GPUs (via SYCL's oneAPI / Level Zero backend)

The examples assume you are familiar with PBS Pro job scheduling and basic compilation workflows at ALCF.

---

## 1. Launch an Interactive Job

⚠️ ALCF uses **PBS Pro** (not SLURM).

### Polaris
```bash
qsub -I -l select=1,walltime=0:60:00,filesystems=home:grand -A alcf_training -q HandsOnHPC
```

### Aurora
```bash
qsub -I -l select=1,walltime=0:60:00,filesystems=home:eagle -A alcf_training -q HandsOnHPC
```

## 2. Modules to load for SYCL

### Polaris
```bash
module use /soft/modulefiles   # Can do module `show` to get more help. DPC++/Clang with SYCL CUDA support
module load PrgEnv-gnu cudatoolkit-standalone/12.9.0 spack-pe-base cmake mpiwrappers/cray-mpich-oneapi-upstream
```

### Aurora
```bash
module use /soft/modulefiles
module restore                 # Intel oneAPI compilers for Aurora (default)
```

## 3. Verify SYCL installation

### Polaris
```bash
$ clang++ -v
clang version 21.0.0git (https://github.com/intel/llvm.git 9375f35d7d0dce54d3d0006da719b71dd682232f)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /soft/compilers/oneapi/upstream/install_09172025/bin
Build config: +assertions
Found candidate GCC installation: /usr/lib64/gcc/x86_64-suse-linux/12
Found candidate GCC installation: /usr/lib64/gcc/x86_64-suse-linux/13
Found candidate GCC installation: /usr/lib64/gcc/x86_64-suse-linux/14
Found candidate GCC installation: /usr/lib64/gcc/x86_64-suse-linux/7
Selected GCC installation: /usr/lib64/gcc/x86_64-suse-linux/14
Candidate multilib: .;@m64
Selected multilib: .;@m64
Found CUDA installation: /soft/compilers/cudatoolkit/cuda-12.9.0, version

$ sycl-ls --verbose
INFO: Output filtered by ONEAPI_DEVICE_SELECTOR environment variable, which is set to cuda:gpu.
To see device ids, use the --ignore-device-selectors CLI option.

<LOADER>[INFO]: loaded adapter 0x0x12bab780 (libur_adapter_cuda.so.0) from /soft/compilers/oneapi/upstream/install_09172025/lib64/libur_adapter_cuda.so.0
[cuda:gpu] NVIDIA CUDA BACKEND, NVIDIA A100-SXM4-40GB 8.0 [CUDA 12.7]
[cuda:gpu] NVIDIA CUDA BACKEND, NVIDIA A100-SXM4-40GB 8.0 [CUDA 12.7]
[cuda:gpu] NVIDIA CUDA BACKEND, NVIDIA A100-SXM4-40GB 8.0 [CUDA 12.7]
[cuda:gpu] NVIDIA CUDA BACKEND, NVIDIA A100-SXM4-40GB 8.0 [CUDA 12.7]
$
```

### Aurora
```bash
$ icpx -v
Intel(R) oneAPI DPC++/C++ Compiler 2025.0.4 (2025.0.4.20241205)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /opt/aurora/24.347.0/oneapi/compiler/2025.0/bin/compiler
Configuration file: /opt/aurora/24.347.0/oneapi/compiler/2025.0/bin/compiler/../icpx.cfg
Found candidate GCC installation: /opt/aurora/24.347.0/spack/unified/0.9.2/install/linux-sles15-x86_64/gcc-13.3.0/gcc-13.3.0-4enwbrb/lib/gcc/x86_64-pc-linux-gnu/13.3.0
Selected GCC installation: /opt/aurora/24.347.0/spack/unified/0.9.2/install/linux-sles15-x86_64/gcc-13.3.0/gcc-13.3.0-4enwbrb/lib/gcc/x86_64-pc-linux-gnu/13.3.0
Candidate multilib: .;@m64
Selected multilib: .;@m64

$sycl-ls --verbose
```


## 4. Compilation for SYCL

### Polaris
```bash
export  CXXFLAGS="-fsycl -fsycl-targets=nvptx64-nvidia-cuda -Xsycl-target-backend --cuda-gpu-arch=sm_80"
```

### Aurora
```bash
export  CXXFLAGS="-fsycl -fsycl-targets=intel_pvc_gpu"
```