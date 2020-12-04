#!/bin/sh
#COBALT -q full-node -A SDL_Workshop -n 1 -t 3:00:00 -O MPI
#Setup the Python environment to include TensorFlow, Keras, PyTorch and Horovod
source /lus/theta-fs0/software/datascience/thetagpu/anaconda3/setup.sh
#
cd cpu
LD_PRELOAD=/lus/theta-fs0/software/datascience/thetagpu/hpctw/lib/libmpitrace.so mpirun -np 8 python ../pytorch_cifar10.py --device cpu >& pytorch_cifar10.out
cd ../gpu
LD_PRELOAD=/lus/theta-fs0/software/datascience/thetagpu/hpctw/lib/libmpitrace.so mpirun -np 8 python ../pytorch_cifar10.py --device gpu >& pytorch_cifar10.out
cd -

