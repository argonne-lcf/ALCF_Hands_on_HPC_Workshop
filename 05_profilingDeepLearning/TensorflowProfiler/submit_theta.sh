#!/bin/bash -l
#COBALT -n 2
#COBALT -t 1:00:00
#COBALT -q training-knl
#COBALT -A SDL_Workshop
#COBALT -O logdir/$COBALT_JOBID

#submisstion script for running tensorflow_mnist with horovod

echo "Running Cobalt Job $COBALT_JOBID."

#Loading modules

module load conda/2021-09-22

PROC_PER_NODE=4
export OMP_NUM_THREADS=32
export KMP_BLOCKTIME=0
export HDF5_USE_FILE_LOCKING='FALSE'
aprun -n $(($COBALT_JOBSIZE*$PROC_PER_NODE)) -N $PROC_PER_NODE -cc none \
    python tensorflow2_cifar10.py  --epochs 1 --logdir logdir/$COBALT_JOBID --num_inter $OMP_NUM_THREADS --num_intra $OMP_NUM_THREADS

