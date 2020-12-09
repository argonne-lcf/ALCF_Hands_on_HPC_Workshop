#!/bin/bash
#COBALT -q training -A SDL_Workshop -n 1 -t 1:00:00 -O MPI_single_node

source /lus/theta-fs0/software/thetagpu/conda/pt_master/2020-11-25/mconda3/setup.sh

export LD_PRELOAD=/lus/theta-fs0/software/datascience/thetagpu/hpctw/lib/libmpitrace.so

[ -e ../../../01_distributedDeepLearning/Horovod/datasets ] && ln -s ../../../01_distributedDeepLearning/Horovod/datasets datasets

mpirun -np 8 python ../pytorch_cifar10.py --device cpu >& pytorch_cifar10.out.cpu


