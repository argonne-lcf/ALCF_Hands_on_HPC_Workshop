#!/bin/bash
#COBALT -n 2
#COBALT -t 1:00:00 -q full-node
#COBALT -A SDL_Workshop -O results/thetagpu/$jobid.pytorch_mnist

#submisstion script for running tensorflow_mnist with horovod

echo "Running Cobalt Job $COBALT_JOBID."

#Loading modules

source /lus/theta-fs0/projects/datascience/parton/thetagpu/pt-build/pt-intall/mconda3/setup.sh

# Single node
for n in 1 2 4 8
do
    mpirun -np $n python pytorch_mnist.py --device gpu --epochs 32 >& results/thetagpu/pytorch_mnist.n$n.out
done
# Go beyond one node
mpirun -np 16 -npernode 8 --hostfile $COBALT_NODEFILE pytorch_mnist.py --device gpu --epochs 32 >& results/thetagpu/pytorch_mnist.n16.out


