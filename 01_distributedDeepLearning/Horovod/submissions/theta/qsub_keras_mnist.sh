#!/bin/bash
#COBALT -n 4
#COBALT -t 1:00:00
#COBALT -q training --attrs mcdram=cache:numa=quad
#COBALT -A SDL_Workshop -O results/theta/$jobid.tensorflow2_keras_mnist

#submisstion script for running tensorflow2_keras_mnist with horovod

echo "Running Cobalt Job $COBALT_JOBID."

#Loading modules

module load datascience/tensorflow-2.3

PROC_PER_NODE=4

aprun -n $(($COBALT_JOBSIZE*$PROC_PER_NODE)) -N $PROC_PER_NODE \
    -j 2 -d 32 -cc depth \
    -e OMP_NUM_THREADS=32 \
    -e KMP_BLOCKTIME=0 \
    python tensorflow2_keras_mnist.py --device cpu --epochs 32 >& results/theta/tensorflow2_keras_mnist.out 

