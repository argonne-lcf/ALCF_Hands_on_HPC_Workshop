#!/bin/bash
#COBALT -n 1
#COBALT -t 1:00:00 
#COBALT -q training-gpu
#COBALT -A SDL_Workshop
#COBALT --attrs=pubnet


echo "Running Cobalt Job $COBALT_JOBID."

if [[ $# -ne 1 ]]; then
    echo "missing argument: use qsub submit_thetagpu.sh <profiler>" >&2
    exit 1
fi

profiler=$1

## load TF
module load conda/tensorflow
conda activate

## set proxies
export https_proxy=http://proxy.tmi.alcf.anl.gov:3128
export http_proxy=http://proxy.tmi.alcf.anl.gov:3128


if [[ $profiler == "nsight" ]]
then
  echo "Using Nsight profiler"

## the following steps can be run
## (A) get flat profile that shows all kernels and time consumed using nsight systems

nsys profile --stats=true --output nsys.out python tensorflow2_cifar10.py --device gpu --epochs 5 > nsys-profile.log 2>&1

## (B) get detailed analysis of CUDA kernels on GPU with Nsight Compute
## (B.1) collect overall compute, memory and I/O utilization
# ncu --kernel-id ::regex:gemm:2  python tensorflow2_cifar10.py --device gpu --epochs 5 > nsight-metrics.log 2>&1

## (B.2) collect metrics for kernels, since it is consumes lot of time for all kernels, here we get metrics for gemm kernels with "--kernel-id" parameter
# ncu --kernel-id ::regex:gemm:2 --metrics all python tensorflow2_cifar10.py --device gpu --epochs 5 > nsight-metrics.log 2>&1


## (C) use GUI 
## (C.1) collect profile output using Nsight systems

# nsys profile --stats=true --output nsys.out python tensorflow2_cifar10.py --device gpu --epochs 5 > nsys-profile.log 2>&1

## steps to visualize results
## (C.2) tunnel using ssh -XL to enable GUI or copy nsys.out.qdrep file to local machine which has Nvidia Nsight UI installed in local machine,
# nsys-ui nsys.out.qdrep


elif [[ $profiler == "dlprof" ]]
then
        echo "Using DLprof profiler"

        pip install --user nvidia-pyindex
        pip install --user nvidia-dlprof

        mkdir ./results
        dlprof --reports=all --mode=simple --nsys_opts="-t cuda,nvtx --force-overwrite true" --output_path=./results/  python tensorflow2_cifar10.py --epochs 1 --device gpu

fi

