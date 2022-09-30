#!/bin/bash

PREFIX="$1"
ENVNAME=ssim

echo set the environment
module swap PrgEnv-intel PrgEnv-gnu
export CRAYPE_LINK_TYPE=dynamic
module unload craype-mic-knl
module load miniconda-3/2021-07-28

echo create the conda environment
conda create -y -c conda-forge --prefix $PREFIX/$ENVNAME python=3.8 pip
conda activate $PREFIX/$ENVNAME
conda install -y -c conda-forge pytorch=1.7.1
conda install -y -c conda-forge matplotlib
conda install -y -c alcf-theta mpi4py
conda install -y -c conda-forge git-lfs
git lfs install

echo install smartsim
git clone https://github.com/CrayLabs/SmartSim.git --depth=1 --branch v0.3.2 smartsim-0.3.2
cd smartsim-0.3.2
pip install -e .[dev,ml]
smart -v --device cpu
cd ..

echo install smartredis
git clone https://github.com/CrayLabs/SmartRedis.git --depth=1 --branch v0.2.0 smartredis-0.2.0
cd smartredis-0.2.0
pip install -e .[dev]

export CC=/opt/gcc/9.3.0/bin/gcc
export CXX=/opt/gcc/9.3.0/bin/g++
make deps
make test-deps
make lib
cd ..
