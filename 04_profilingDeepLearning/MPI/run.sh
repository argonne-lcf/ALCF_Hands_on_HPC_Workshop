#!/bin/sh
#Setup the Python environment to include TensorFlow, Keras, PyTorch and Horovod
source /lus/theta-fs0/software/datascience/thetagpu/anaconda3/setup.sh
#
LD_PRELOAD=/lus/theta-fs0/software/datascience/thetagpu/hpctw/lib/libmpitrace.so mpirun -np 4 python pytorch_cifar10.py --device cpu 
LD_PRELOAD=/lus/theta-fs0/software/datascience/thetagpu/hpctw/lib/libmpitrace.so mpirun -np 8 python pytorch_cifar10.py --device gpu

