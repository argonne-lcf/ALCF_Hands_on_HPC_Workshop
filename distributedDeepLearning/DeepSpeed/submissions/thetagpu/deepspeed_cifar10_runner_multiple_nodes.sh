#!/bin/bash

echo "Setting up env"

conda env create --name deepspeed --file /lus/theta-fs0/projects/datascience/zhen/env_deepspeed.yml

conda activate deepspeed

cd /lus/theta-fs0/projects/datascience/zhen/DeepSpeed
 
echo "Current directory: "
pwd
 
echo "Run script: "
deepspeed --hostfile=$COBALT_NODEFILE cifar10_deepspeed.py --deepspeed --deepspeed_config ds_config.json $@

