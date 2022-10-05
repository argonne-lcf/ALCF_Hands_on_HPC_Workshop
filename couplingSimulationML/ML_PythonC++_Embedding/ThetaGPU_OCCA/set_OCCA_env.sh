#!/bin/bash

module load cmake
module load conda/2021-11-30

SDL=sdl_workshop/05_Simulation_ML/ML_PythonC++_Embedding/ThetaGPU_OCCA/

export OCCA_DIR=/lus/grand/projects/catalyst/world-shared/spatel/occa/install
export OCCA_CACHE_DIR=/path/to/$SDL
#export OCCA_CACHE_DIR=/lus/grand/projects/catalyst/spatel/$SDL

export PATH+=":${OCCA_DIR}/bin"
export LD_LIBRARY_PATH+=":${OCCA_DIR}/lib"

export OCCA_CXX="g++"
export OCCA_CXXFLAGS="-O3"

export OCCA_CUDA_COMPILER="nvcc"
export OCCA_CUDA_COMPILER_FLAGS="-O3 --fmad=true"
