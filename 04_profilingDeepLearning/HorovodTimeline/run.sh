#!/bin/sh
#COBALT -q full-node -A SDL_Workshop -n 1 -t 3:00:00 -O HorovodTimeline
#Setup the Python environment to include TensorFlow, Keras, PyTorch and Horovod
source /lus/theta-fs0/software/datascience/thetagpu/anaconda3/setup.sh
#
cd cpu
HOROVOD_TIMELINE=cpu_timeline.json mpirun -np 8 python ../pytorch_cifar10.py --device cpu >& pytorch_cifar10.out
cd ../gpu
HOROVOD_TIMELINE=gpu_timeline.json mpirun -np 8 python ../pytorch_cifar10.py --device gpu >& pytorch_cifar10.out
cd -

