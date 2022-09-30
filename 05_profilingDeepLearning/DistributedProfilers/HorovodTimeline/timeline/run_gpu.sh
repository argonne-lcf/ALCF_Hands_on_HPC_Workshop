#!/bin/bash

module load conda/2021-09-22
conda activate

NODES=$1

export HOROVOD_TIMELINE=timeline_gpu_n${NODES}.json
export HOROVOD_TIMELINE_MARK_CYCLES='1'

mpirun  -x HOROVOD_TIMELINE -x HOROVOD_TIMELINE_MARK_CYCLES -x LD_LIBRARY_PATH -x PYTHONPATH -x PATH  --hostfile ${COBALT_NODEFILE}\
       -n $NODES  -npernode 8 python ../../tensorflow2_mnist.py --device gpu >& gpu_n${NODES}.out


