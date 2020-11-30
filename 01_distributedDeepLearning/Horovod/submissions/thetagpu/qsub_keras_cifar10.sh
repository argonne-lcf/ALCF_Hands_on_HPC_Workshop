#!/bin/bash
#COBALT -n 1
#COBALT -t 1:00:00 -q full-node
#COBALT -A SDL_Workshop -O keras_cifar10

#submisstion script for running tensorflow_mnist with horovod

echo "Running Cobalt Job $COBALT_JOBID."

#Loading modules

source /lus/theta-fs0/software/datascience/thetagpu/anaconda3/setup.sh

for n in 1 2 4 8
do
    mpirun -np $n python tensorflow2_keras_cifar10.py --device gpu --epochs 32 >& tensorflow2_keras_cifar10.out.$n
done

