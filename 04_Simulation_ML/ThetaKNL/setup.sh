#!/bin/sh
unset PYTHONPATH

module use /soft/datascience/a21/modulefiles
module load intelpython36

TF_DIR=/soft/datascience/tensorflow/tf2.2-py36
TORCH_DIR=/soft/datascience/pytorch/1.4.0-py36/lib/python3.6/site-packages

export PYTHONPATH=${TF_DIR}:${TORCH_DIR}:$PYTHONPATH
export LD_LIBRARY_PATH=/soft/datascience/a21/oneccl/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/cray/pe/mpt/7.7.14/gni/mpich-intel-abi/16.0/lib:$LD_LIBRARY_PATH

# Clean up the directory
rm -rf *.png
cd build/

rm -rf *.png
rm -rf *.npy
rm -rf checkpoints/
rm -rf __pycache__/
rm -rf CMakeFiles/
rm -rf app
rm -rf cmake_install.cmake
rm -rf Makefile
rm -rf CMakeCache.txt
cd ..
