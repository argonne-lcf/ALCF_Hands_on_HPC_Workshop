#!/bin/bash
#COBALT -n 1
#COBALT -t 0:10:00 -q full-node
#COBALT -A SDL_Workshop
#COBALT --attrs=pubnet
#COBALT -O cifar10_1node_4gpus

#submisstion script for running pytorch_mnist with DDP

echo "Running Cobalt Job $COBALT_JOBID."

# Which container will we use?
CONTAINER=/lus/theta-fs0/software/thetagpu/nvidia-containers/pytorch/pytorch_20.11-py3.simg
SCRIPT=/home/cadams/ThetaGPU/sdl_ai_workshop/01_distributedDeepLearning/DDP/submissions/thetagpu/pytorch_cifar10_runner.sh


N_NODES=$(cat $COBALT_NODEFILE | wc -l)
RANKS_PER_NODE=4
let N_RANKS=${RANKS_PER_NODE}*${N_NODES}



# Here's the MPI Command, using a heredoc to encapsulate the venv setup:
mpirun -n $N_RANKS -hostfile ${COBALT_NODEFILE} -map-by node singularity run --nv -B /lus:/lus $CONTAINER $SCRIPT
