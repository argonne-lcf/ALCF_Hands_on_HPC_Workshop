#!/bin/bash
#COBALT -n 4
#COBALT -t 1:00:00
#COBALT -q ATPESC2020 --attrs mcdram=cache:numa=quad
#COBALT -A ATPESC2020 -O pytorch_mnist

#submisstion script for running tensorflow_mnist with horovod

echo "Running Cobalt Job $COBALT_JOBID."

#Loading modules

source /projects/ATPESC2020/hzheng/ATPESC_MachineLearning/DataParallelDeepLearning/setup.sh

PROC_PER_NODE=4

aprun -n $(($COBALT_JOBSIZE*$PROC_PER_NODE)) -N $PROC_PER_NODE \
    -j 2 -d 32 -cc depth \
    -e OMP_NUM_THREADS=32 \
    -e KMP_BLOCKTIME=0 \
    python pytorch_mnist.py --num_threads=32 --device cpu --epochs 8

