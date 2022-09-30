#!/bin/bash

echo "Entering the container"
echo "Setting up venv"
source /lus/theta-fs0/software/thetagpu/nvidia-containers/pytorch/venvs/20.11/bin/activate
echo "Which python?"
which python

cd /home/cadams/ThetaGPU/sdl_ai_workshop/01_distributedDeepLearning/DDP/
echo "Current Directory: "
pwd

python pytorch_mnist.py --device gpu --epochs 32
