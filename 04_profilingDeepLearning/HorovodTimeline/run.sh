#!/bin/sh
#Setup the Python environment to include TensorFlow, Keras, PyTorch and Horovod
source /lus/theta-fs0/software/datascience/thetagpu/anaconda3/setup.sh
#
HOROVOD_TIMELINE=cpu_timeline.json mpirun -np 4 python pytorch_cifar10.py --device cpu 
HOROVOD_TIMELINE=gpu_timeline.json mpirun -np 8 python pytorch_cifar10.py --device gpu

