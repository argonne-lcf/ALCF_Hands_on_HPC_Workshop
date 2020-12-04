#!/bin/bash
#COBALT -n 2
#COBALT -t 1:00:00 -q full-node
#COBALT -A SDL_Workshop -O results/thetagpu/$jobid.pytorch_cifar10

#submisstion script for running tensorflow_mnist with horovod

echo "Running Cobalt Job $COBALT_JOBID."

#Loading modules

source /lus/theta-fs0/projects/datascience/parton/thetagpu/pt-build/pt-intall/mconda3/setup.sh

mpirun -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -np 16 -npernode 8 --hostfile $COBALT_NODEFILE pytorch_cifar10.py --device gpu --epochs 64 >& results/thetagpu/pytorch_cifar10.n16.out

# for doing a scaling test
#for n in 1 2 4 8
#do
#    mpirun -np $n python pytorch_cifar10.py --device gpu --epochs 64 >& results/thetagpu/pytorch_cifar10.n$n.out
#done

