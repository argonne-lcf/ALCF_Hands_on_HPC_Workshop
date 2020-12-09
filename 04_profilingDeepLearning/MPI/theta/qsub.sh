#!/bin/bash
#COBALT -q training -A SDL_Workshop -n 32 -t 0:20:00 -O MPI

#Setup the Python environment to include TensorFlow, Keras, PyTorch and Horovod

module load datascience/pytorch-1.7

export LD_PRELOAD=/soft/perftools/hpctw/NONE/libhpmprof.so

#[ -e ../../../01_distributedDeepLearning/Horovod/datasets ] && ln -s ../../../01_distributedDeepLearning/Horovod/datasets datasets

PROC_PER_NODE=4
aprun -n $(($COBALT_JOBSIZE*$PROC_PER_NODE)) -N $PROC_PER_NODE \
      -j 2 -d 32 -cc depth \
      -e OMP_NUM_THREADS=32 \
      -e KMP_BLOCKTIME=0 \
      python ../pytorch_cifar10.py --num_threads=32 --device cpu --epochs 5

