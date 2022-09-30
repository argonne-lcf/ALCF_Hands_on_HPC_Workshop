#!/bin/bash -l
#COBALT -n 1
#COBALT -t 10
#COBALT -q training-gpu
#COBALT -A SDL_Workshop
#COBALT -O logdir/$COBALT_JOBID

#submisstion script for running tensorflow_mnist with horovod

echo "Running Cobalt Job $COBALT_JOBID."

# Loading conda environment with Tensorflow
module load conda/tensorflow
conda activate

export OMP_NUM_THREADS=64
n=8
mpirun -np $n python tensorflow2_cifar10.py  --epochs 1 --logdir logdir/$COBALT_JOBID --num_inter $OMP_NUM_THREADS --num_intra $OMP_NUM_THREADS

