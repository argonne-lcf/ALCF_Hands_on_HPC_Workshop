#!/bin/bash
#COBALT -n 2
#COBALT -t 0:10:00 -q full-node
#COBALT -A SDL_Workshop
#COBALT --attrs=pubnet
#COBALT -O cifar10_1node_4gpus

#submisstion script for running cifar10 with deepspeed

echo "Running Cobalt Job $COBALT_JOBID."

echo "Setting up env"

conda env create --name deepspeed --file /lus/theta-fs0/projects/datascience/zhen/env_deepspeed.yml

conda activate deepspeed

cd /lus/theta-fs0/projects/datascience/zhen/DeepSpeed
 
echo "Current directory: "
pwd
 
echo "Run script: "
deepspeed --hostfile=$COBALT_NODEFILE cifar10_deepspeed.py --deepspeed --deepspeed_config ds_config.json $@
