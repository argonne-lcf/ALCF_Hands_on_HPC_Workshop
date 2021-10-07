#!/bin/bash

cd /home/$USER/sdl_ai_workshop/05_Simulation_ML/ML_PythonC++_Embedding/ThetaGPU
echo $pwd

# Clean up the directory
rm -rf *.png
cd app_build/

rm -rf *.png
rm -rf *.npy
rm -rf checkpoints/
rm -rf __pycache__/
rm -rf CMakeFiles/
rm -rf app
rm -rf cmake_install.cmake
rm -rf Makefile
rm -rf CMakeCache.txt

export VENV_LOCATION=/home/$USER/THETAGPU_TF_ENV
source $VENV_LOCATION/bin/activate

cmake ../
make

./app
