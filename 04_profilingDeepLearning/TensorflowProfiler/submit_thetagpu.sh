#!/bin/bash
#COBALT -n 1
#COBALT -t 10
#COBALT -q training
#COBALT -A SDL_Workshop
#COBALT -O logdir/$COBALT_JOBID

#submisstion script for running tensorflow_mnist with horovod

echo "Running Cobalt Job $COBALT_JOBID."

# Loading conda environment with Tensorflow
source /lus/theta-fs0/software/thetagpu/conda/tf_master/2020-11-11/mconda3/setup.sh

export OMP_NUM_THREADS=64
n=8
mpirun -np $n python tensorflow2_cifar10.py  --epochs 1 --logdir logdir/$COBALT_JOBID --num_inter $OMP_NUM_THREADS --num_intra $OMP_NUM_THREADS

