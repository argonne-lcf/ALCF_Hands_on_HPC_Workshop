#!/bin/bash

module load conda/2021-09-22
conda activate

NODES=$1

export HOROVOD_AUTOTUNE_LOG="horovod_tune.log"
export HOROVOD_AUTOTUNE=1


mpirun -x HOROVOD_AUTOTUNE -x HOROVOD_AUTOTUNE_LOG -x LD_LIBRARY_PATH -x PYTHONPATH -x PATH  --hostfile ${COBALT_NODEFILE}\
       -n $NODES  -npernode 8 python ../../tensorflow2_mnist.py --device gpu --epochs 500 >& gpu_n${NODES}_tune.out

