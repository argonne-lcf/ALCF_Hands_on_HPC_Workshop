#!/bin/bash
#COBALT -n 1
#COBALT -t 10
#COBALT -A SDL_Workshop

#submisstion script for running tensorflow_mnist with horovod

echo "Running Cobalt Job $COBALT_JOBID."

# Loading conda environment with Tensorflow
source /lus/theta-fs0/software/thetagpu/conda/tf_master/latest/mconda3/setup.sh

n=8
mpirun -np $n python tensorflow2_cifar10.py --device gpu --epochs 1 >& ${COBALT_JOBID}.tensorflow2_cifar10.n$n.out

