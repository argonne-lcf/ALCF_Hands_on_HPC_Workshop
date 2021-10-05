#!/bin/bash
#COBALT -q training -A SDL_Workshop  -t 10:00 -O HorovodTimeline

module load conda/2021-09-22
conda activate

NODES=$(cat $COBALT_NODEFILE | wc -l)

export HOROVOD_TIMELINE=timeline_cpu_n${NODES}.json


CUDA_VISIBLE_DEVICES="" mpirun  -x HOROVOD_TIMELINE -x LD_LIBRARY_PATH -x PYTHONPATH -x PATH  --hostfile ${COBALT_NODEFILE} \
       -n $NODES  -npernode 8 python ../../tensorflow2_mnist.py --device cpu >& cpu_n${NODES}.out

