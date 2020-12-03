#!/bin/bash
#COBALT -n 2
#COBALT -t 1:00:00 -q full-node
#COBALT -A SDL_Workshop -O results/thetagpu/$jobid.keras_cifar10

#submisstion script for running tensorflow_mnist with horovod

echo "Running Cobalt Job $COBALT_JOBID."

#Loading modules

source /lus/theta-fs0/software/thetagpu/conda/tf_master/latest/mconda3/setup.sh

mpirun -np 16 -npernode 8 --hostfile ${COBALT_NODEFILE} $(which python) tensorflow2_keras_cifar10.py --device gpu --epochs 32 >& results/thetagpu/tensorflow2_keras_cifar10.out.16

#for n in 1 2 4 8
#do
#    mpirun -np $n $(which python) tensorflow2_keras_cifar10.py --device gpu --epochs 32 >& results/thetagpu/tensorflow2_keras_cifar10.out.$n
#done

