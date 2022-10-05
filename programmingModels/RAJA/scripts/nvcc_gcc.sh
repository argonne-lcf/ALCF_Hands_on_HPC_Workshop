#!/bin/bash

BUILD_SUFFIX=alcf_nvcc_gcc

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}


#module load cmake

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ../host-configs/alcf-builds/nvcc_gcc.cmake \
  -DENABLE_OPENMP=Off \
  -DENABLE_CUDA=On \
  -DCMAKE_CUDA_ARCHITECTURES="80" \
  -DCMAKE_INSTALL_PREFIX=../install_${BUILD_SUFFIX} \
  "$@" \
  ..
