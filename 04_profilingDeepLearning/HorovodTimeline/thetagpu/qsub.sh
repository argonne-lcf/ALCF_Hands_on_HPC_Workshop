#!/bin/bash
#COBALT -q training -A SDL_Workshop -n 1 -t 1:00:00 -O HorovodTimeline

source /lus/theta-fs0/software/thetagpu/conda/pt_master/2020-11-25/mconda3/setup.sh
COBALT_JOBSIZE=$(cat $COBALT_NODEFILE | wc -l)

export HOROVOD_TIMELINE=thetagpu_timeline_n${COBALT_JOBSIZE}.json

[ -e ../../../01_distributedDeepLearning/Horovod/datasets ] && ln -s ../../../01_distributedDeepLearning/Horovod/datasets datasets

mpirun -x HOROVOD_TIMELINE -x LD_LIBRARY_PATH -x PYTHONPATH -x PATH --hostfile ${COBALT_NODEFILE} \
       -np $((COBALT_JOBSIZE*8)) -npernode 8 python ../pytorch_cifar10.py --device gpu >& pytorch_cifar10.out.${COBALT_JOBSIZE}


