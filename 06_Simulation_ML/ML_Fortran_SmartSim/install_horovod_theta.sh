#!/bin/bash

### Install Horovod
# NOTE: - need to source the SSIM conda environment first

# set the environment
#module swap PrgEnv-intel PrgEnv-gnu
module swap gcc gcc/8.3.0
export CRAY_CPU_TARGET=mic-knl

# Horovod source and version
HOROVOD_REPO_URL=https://github.com/uber/horovod.git
HOROVOD_REPO_TAG=v0.21.3

echo Clone Horovod $HOROVOD_REPO_TAG git repo
DIR_NAME=horovod-$HOROVOD_REPO_TAG
git clone --recursive $HOROVOD_REPO_URL $DIR_NAME
cd $DIR_NAME
git checkout $HOROVOD_REPO_TAG

HOROVOD_CMAKE=$(which cmake) HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 CC=$(which cc) CXX=$(which CC) python setup.py bdist_wheel
HVD_WHL=$(find dist/ -name "horovod*.whl" -type f)
cp $HVD_WHL .
HVD_WHEEL=$(find . -maxdepth 1 -name "horovod*.whl" -type f)
echo Install Horovod $HVD_WHEEL
pip install --force-reinstall $HVD_WHEEL
