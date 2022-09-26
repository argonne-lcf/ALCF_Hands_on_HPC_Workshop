#!/bin/bash

module load conda/2021-09-22
conda activate

export LD_PRELOAD=/lus/theta-fs0/software/datascience/thetagpu/hpctw/lib/libmpitrace.so

NPROC=$1

CUDA_VISIBLE_DEVICES="" mpirun -n $NPROC python ../tensorflow2_mnist.py --device cpu --epochs 4  >& cpu_n$NPROC.out


