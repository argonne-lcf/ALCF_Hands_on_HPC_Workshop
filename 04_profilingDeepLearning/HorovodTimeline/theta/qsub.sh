#!/bin/bash
#COBALT -q training -A SDL_Workshop -n 8 -t 1:00:00 -O HorovodTimeline

# Note that you have to submit this jobs on Theta login node (or mom node)

#Setup the Python environment to include TensorFlow, Keras, PyTorch and Horovod

module load datascience/pytorch-1.7
# Copy dataset, so that we don't need to redownload them again

[ -e ../../../01_distributedDeepLearning/Horovod/datasets ] && ln -s ../../../01_distributedDeepLearning/Horovod/datasets datasets

PROC_PER_NODE=4
export HOROVOD_TIMELINE=theta_timeline_${COBALT_JOBSIZE}.json
aprun -n $(($COBALT_JOBSIZE*$PROC_PER_NODE)) -N $PROC_PER_NODE \
      -j 2 -d 32 -cc depth \
      -e OMP_NUM_THREADS=32 \
      -e KMP_BLOCKTIME=0 \
      python ../pytorch_cifar10.py --num_threads=32 --device cpu --epochs 4

