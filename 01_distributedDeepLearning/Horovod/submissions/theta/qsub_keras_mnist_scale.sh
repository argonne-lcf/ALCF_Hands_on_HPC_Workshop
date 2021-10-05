#!/bin/bash
#COBALT -n 128
#COBALT -t 1:00:00
#COBALT -q training-knl --attrs mcdram=cache:numa=quad
#COBALT -A SDL_Workshop -O results/theta/$jobid.tensorflow2_keras_mnist_scale

#submisstion script for running tensorflow2_keras_mnist with horovod

echo "Running Cobalt Job $COBALT_JOBID."

#Loading modules

module load datascience/tensorflow-2.3

PROC_PER_NODE=4
for n in 1 2 4 8 16 32 64 
do
    aprun -n $((n*$PROC_PER_NODE)) -N $PROC_PER_NODE \
	  -j 2 -d 32 -cc depth \
	  -e OMP_NUM_THREADS=32 \
	  -e KMP_BLOCKTIME=0 \
	  python tensorflow2_keras_mnist.py --device cpu --epochs 32 >& results/theta/tensorflow2_keras_mnist.out.$n & 
done
wait


for n in 1 2 4 8 16 32 64
do
    aprun -n $((n*$PROC_PER_NODE)) -N $PROC_PER_NODE \
	  -j 2 -d 32 -cc depth \
	  -e OMP_NUM_THREADS=32 \
	  -e KMP_BLOCKTIME=0 \
	  python tensorflow2_keras_mnist.py --device cpu --epochs 32 --warmup_epochs 0 >& results/theta/tensorflow2_keras_mnist.out.${n}-nowarmup & 
done
wait

