#!/bin/bash -e

ENV_PREFIX=$PWD

## Load conda module and activate base env
module load conda/2022-09-08
conda activate base
module load cmake

## Create a new conda env at specified path
conda create -p $ENV_PREFIX/ssim python=3.9 -y
conda activate $ENV_PREFIX/ssim

## Set some env variables
#export CRAYPE_LINK_TYPE=dynamic  # set by default
export SMARTSIM_REDISAI=1.2.7
export CC=cc
export CXX=CC

## Clone and install SmartSim
git clone https://github.com/CrayLabs/SmartSim.git --branch v0.4.2
cd SmartSim
pip install -e .[ml] # The [ml] extension installs TensorFlow 2.8
cd ..

## Install GPU backend (RedisAI and PyTorch)
export CUDNN_LIBRARY=/soft/libraries/cudnn/cudnn-11.5-linux-x64-v8.3.3.40/lib/
export CUDNN_INCLUDE_DIR=/soft/libraries/cudnn/cudnn-11.5-linux-x64-v8.3.3.40/include/
export LD_LIBRARY_PATH=$CUDNN_LIBRARY:$LD_LIBRARY_PATH
cd SmartSim
# This installs torch==1.11.0+cu113 torchvision==0.12.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
smart build -v --device gpu | tee build_backend.log
# Upgrate CUDA lib version to 11.5 so it is compatible with CUDA needed for HVD build
pip install torch==1.11.0+cu115 torchvision==0.12.0+cu115 -f https://download.pytorch.org/whl/torch_stable.html
cd ..

## Install SmartRedis Client Library
git clone https://github.com/CrayLabs/SmartRedis.git --branch v0.4.1
cd SmartRedis
#pip install -e .
make lib
cd ..

## Install KeyDB
#export CPATH=~/include # needed for curl untill installed on Polaris
#export LIBRARY_PATH=~/lib # needed for curl until installed on Polaris
#git clone https://www.github.com/eq-alpha/keydb.git --branch v6.2.0
#cd keydb
#CC=gcc CXX=g++ make -j 8
#cd ..

## Install Horovod
HOROVOD_REPO_TAG="v0.25.0"
HOROVOD_REPO_URL=https://github.com/uber/horovod.git

CUDA_VERSION_MAJOR=11
CUDA_VERSION_MINOR=5
CUDA_VERSION_MINI=2
CUDA_VERSION_BUILD=495.29.05
CUDA_VERSION=$CUDA_VERSION_MAJOR.$CUDA_VERSION_MINOR
CUDA_VERSION_FULL=$CUDA_VERSION.$CUDA_VERSION_MINI
CUDA_TOOLKIT_BASE=/soft/compilers/cudatoolkit/cuda_${CUDA_VERSION_FULL}_${CUDA_VERSION_BUILD}_linux

CUDA_DEPS_BASE=/soft/libraries/

CUDNN_VERSION_MAJOR=8
CUDNN_VERSION_MINOR=3
CUDNN_VERSION_EXTRA=3.40
CUDNN_VERSION=$CUDNN_VERSION_MAJOR.$CUDNN_VERSION_MINOR.$CUDNN_VERSION_EXTRA
CUDNN_BASE=$CUDA_DEPS_BASE/cudnn/cudnn-$CUDA_VERSION-linux-x64-v$CUDNN_VERSION

NCCL_VERSION_MAJOR=2
NCCL_VERSION_MINOR=12.10-1
NCCL_VERSION=$NCCL_VERSION_MAJOR.$NCCL_VERSION_MINOR
NCCL_BASE=$CUDA_DEPS_BASE/nccl/nccl_$NCCL_VERSION+cuda${CUDA_VERSION}_x86_64

TENSORRT_VERSION_MAJOR=8
TENSORRT_VERSION_MINOR=2.5.1
TENSORRT_VERSION=$TENSORRT_VERSION_MAJOR.$TENSORRT_VERSION_MINOR
TENSORRT_BASE=$CUDA_DEPS_BASE/trt/TensorRT-$TENSORRT_VERSION.Linux.x86_64-gnu.cuda-$CUDA_VERSION.cudnn$CUDNN_VERSION_MAJOR.$CUDNN_VERSION_MINOR

git clone --recursive $HOROVOD_REPO_URL
cd horovod

if [[ -z "$HOROVOD_REPO_TAG" ]]; then
    echo Checkout Horovod master
else
    echo Checkout Horovod tag $HOROVOD_REPO_TAG
    git checkout --recurse-submodules $HOROVOD_REPO_TAG
fi

echo Build Horovod Wheel using MPI from $MPICH_DIR and NCCL from ${NCCL_BASE}
echo "MPI_ROOT=$MPICH_DIR HOROVOD_WITH_MPI=1 HOROVOD_CUDA_HOME=${CUDA_TOOLKIT_BASE} HOROVOD_NCCL_HOME=$NCCL_BASE HOROVOD_CMAKE=$(which cmake) HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 python setup.py bdist_wheel"
MPI_ROOT=$MPICH_DIR HOROVOD_WITH_MPI=1 HOROVOD_CUDA_HOME=${CUDA_TOOLKIT_BASE} HOROVOD_NCCL_HOME=$NCCL_BASE HOROVOD_CMAKE=$(which cmake) HOROVOD_GPU_OPERATIONS=NCCL HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITH_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 python setup.py bdist_wheel

HVD_WHL=$(find dist/ -name "horovod*.whl" -type f)
cp $HVD_WHL .
HVD_WHEEL=$(find . -maxdepth 1 -name "horovod*.whl" -type f)
echo Install Horovod $HVD_WHEEL
pip install --force-reinstall --no-cache-dir $HVD_WHEEL

cd ..

## Install MPI4PY
MPICC="cc -shared" pip install --force-reinstall --no-cache-dir --no-binary=mpi4py mpi4py


## Install any other packages
pip install hydra-core --upgrade


