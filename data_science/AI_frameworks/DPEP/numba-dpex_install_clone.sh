#!/bin/bash

module load frameworks

ENV_PREFIX_PATH=$PWD

# Create a new conda environment
conda create --prefix $ENV_PREFIX_PATH/dpep_env/dpep_env --clone /opt/aurora/24.347.0/frameworks/aurora_nre_models_frameworks-2025.0.0
conda activate $ENV_PREFIX_PATH/dpep_env

# Install numba-dpex and its dependencies
conda install -y scikit-build numba==0.59* -c conda-forge
pip install versioneer
git clone https://github.com/IntelPython/numba-dpex.git
cd numba-dpex
CXX=$(which dpcpp) python setup.py develop
