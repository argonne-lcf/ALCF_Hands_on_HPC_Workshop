#!/bin/bash
#COBALT -n 2
#COBALT -t 1:00:00 -q full-node
#COBALT -A SDL_Workshop -O results/thetagpu/$jobid.tensorflow2_cifar10

#submisstion script for running tensorflow_mnist with horovod

echo "Running Cobalt Job $COBALT_JOBID."

#Loading modules

#source /lus/theta-fs0/software/datascience/thetagpu/anaconda3/setup.sh
source /lus/theta-fs0/software/thetagpu/conda/tf_master/2020-11-11/mconda3/setup.sh
#source /lus/theta-fs0/software/thetagpu/conda/tf_master/latest/mconda3/setup.sh

COBALT_JOBSIZE=$(cat $COBALT_NODEFILE | wc -l)

echo "Running job on ${COBALT_JOBSIZE} nodes"
ng=$((COBALT_JOBSIZE*8))
if (( ${COBALT_JOBSIZE} > 1 ))
then
    # multiple nodes
    mpirun -x LD_LIBRARY_PATH -x PATH -x PYTHONPATH -np $ng -npernode 8 --hostfile $COBALT_NODEFILE python tensorflow2_cifar10.py --device gpu --epochs 32 >& results/thetagpu/tensorflow2_cifar10.n$ng.out
else
    # Single node
    for n in 1 2 4 8
    do
	mpirun -np $n python tensorflow2_cifar10.py --device gpu --epochs 32 >& results/thetagpu/tensorflow2_cifar10.n$n.out
    done
fi



