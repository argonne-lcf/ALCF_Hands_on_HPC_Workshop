#!/bin/bash

module load frameworks
module load cmake

ENV_PREFIX_PATH=$PWD

# Create a new conda environment
conda create -y --prefix $ENV_PREFIX_PATH/dpep_env python=3.10 pip
conda activate $ENV_PREFIX_PATH/dpep_env

# Install dpnp and dpctl
conda install -y -c https://software.repos.intel.com/python/conda/linux-64 -c conda-forge --strict-channel-priority dpctl==0.18.3 dpnp==0.16.3

# Install numba-dpex and its dependencies
conda install -y scikit-build numba==0.59* -c conda-forge
pip install versioneer
git clone https://github.com/IntelPython/numba-dpex.git
cd numba-dpex
CXX=$(which dpcpp) python setup.py develop
