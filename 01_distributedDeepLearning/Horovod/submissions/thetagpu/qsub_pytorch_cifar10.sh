#!/bin/bash
#COBALT -n 1
#COBALT -t 1:00:00 -q full-node
#COBALT -A SDL_Workshop -O pytorch_cifar10

#submisstion script for running tensorflow_mnist with horovod

echo "Running Cobalt Job $COBALT_JOBID."

#Loading modules

source /lus/theta-fs0/software/datascience/thetagpu/anaconda3/setup.sh

for n in 1 2 4 8
do
    mpirun -np $n python pytorch_cifar10.py --device gpu --epochs 256 >& pytorch_cifar10.n$n.out
done

