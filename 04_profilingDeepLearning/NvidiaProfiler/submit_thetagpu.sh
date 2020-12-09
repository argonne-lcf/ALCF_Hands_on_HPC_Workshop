#!/bin/bash
#COBALT -n 1
#COBALT -t 1:00:00 
#COBALT -q training
#COBALT -A SDL_Workshop
#COBALT --attrs=pubnet


echo "Running Cobalt Job $COBALT_JOBID."

## container
IMG=/lus/theta-fs0/software/thetagpu/nvidia-containers/tensorflow2/tf2_20.08-py3.sif

## set library path
export SINGULARITYENV_LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

## set proxies
export https_proxy=http://proxy.tmi.alcf.anl.gov:3128
export http_proxy=http://proxy.tmi.alcf.anl.gov:3128

## the following steps can be run
## (A) get flat profile that shows all kernels and time consumed using nsight systems

singularity exec -B /lus:/lus --nv ${IMG} nsys profile --stats=true --output nsys.out python tensorflow2_cifar10.py --device gpu --epochs 5 > nsys-profile.log 2>&1

## (B) get detailed analysis of CUDA kernels on GPU with Nsight Compute
## (B.1) collect overall compute, memory and I/O utilization
#singularity exec -B /lus:/lus --nv ${IMG} ncu python tensorflow2_cifar10.py --device gpu --epochs 5 > gpu-utilization.log 2>&1

## (B.2) collect metrics for kernels, since it is consumes lot of time for all kernels, here we get metrics for gemm kernels with "--kernel-id" parameter
#singularity exec -B /lus:/lus --nv ${IMG} ncu --kernel-id ::regex:gemm:2 --metrics all python tensorflow2_cifar10.py --device gpu --epochs 5 > nsight-metrics.log 2>&1


## (C) use GUI 
## (C.1) collect profile output using Nsight systems
#singularity exec -B /lus:/lus --nv ${IMG} nsys profile --stats=true --output nsys.out python tensorflow2_cifar10.py --device gpu --epochs 5 > nsys-profile.log 2>&1

## steps to visualize results
## (C.2) tunnel using ssh -XL to enable GUI or copy nsys.out.qdrep file to local machine which has Nvidia Nsight UI installed in local machine,
# nsys-ui nsys.out.qdrep

