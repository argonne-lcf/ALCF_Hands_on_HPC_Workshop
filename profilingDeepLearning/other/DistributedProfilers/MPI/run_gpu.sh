#!/bin/bash

module load conda/2021-09-22
conda activate

export LD_PRELOAD=/lus/theta-fs0/software/datascience/thetagpu/hpctw/lib/libmpitrace.so

NPROC=$1

mpirun -n $NPROC python ../tensorflow2_mnist.py --device gpu  >& gpu_n$NPROC.out


