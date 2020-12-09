#!/bin/bash
#COBALT -n 1
#COBALT -t 1:00:00
#COBALT -q training --attrs mcdram=cache:numa=quad
#COBALT -A SDL_Workshop

echo "Running Cobalt Job $COBALT_JOBID."

#Load modules
module load datascience/pytorch-1.4
module load vtune

## Set libraries
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/soft/compilers/intel/19.0.3.199/vtune_amplifier/lib64/

## set proxies
export https_proxy=http://proxy.tmi.alcf.anl.gov:3128
export http_proxy=http://proxy.tmi.alcf.anl.gov:3128

## run the job
PROC_PER_NODE=1
#aprun -n $(($COBALT_JOBSIZE*$PROC_PER_NODE)) -N $PROC_PER_NODE python pytorch_cifar10.py --device cpu --epochs 5

## use Intel Vtune APS to get application performance snapshot
aprun -n $(($COBALT_JOBSIZE*$PROC_PER_NODE)) -N $PROC_PER_NODE aps python pytorch_cifar10.py --device cpu --epochs 5
aps --report=./aps_result_20201202

## Use Intel Vtune to extract hotspots
aprun -n $(($COBALT_JOBSIZE*$PROC_PER_NODE)) -N $PROC_PER_NODE  amplxe-cl -c hotspots -finalization-mode=none -knob sampling-mode=hw -r vtune-result-dir_advancedhotspots -strategy ldconfig:notrace:notrace --  python pytorch_cifar10.py --device cpu --epochs 1

## (B) From login node
# amplxe-cl -finalize -search-dir / -r vtune-result-dir_advancedhotspots

## steps to visualize results
## (C) tunnel with ssh -XL to run GUI or copy to local machine
# amplxe-gui  vtune-result-dir_advancedhotspots

