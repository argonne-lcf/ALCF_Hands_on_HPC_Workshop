#!/bin/bash
#COBALT -n 2
#COBALT -t 1:00:00
#COBALT -q default 
#COBALT -A datascience -O keras_mnist

#submisstion script for running tensorflow_mnist with horovod

echo "Running Cobalt Job $COBALT_JOBID."

#Loading modules

export PATH=/soft/datascience/anaconda3/bin:$PATH
export PATH=/soft/libraries/mpi/mvapich2/gcc/bin/:$PATH
export PATH=/soft/visualization/cuda-9.0.176/bin:$PATH
export LD_LIBRARY_PATH=/soft/visualization/cuda-9.0.176/lib64/:$LD_LIBRARY_PATH

PROC_PER_NODE=2

mpirun -np 4 -ppn 2 python keras_mnist.py --device gpu

