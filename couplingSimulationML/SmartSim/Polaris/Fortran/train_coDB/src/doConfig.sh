#!/bin/bash

CC=cc
CXX=CC
FC=ftn

cmake \
-DCMAKE_Fortran_FLAGS="-g" \
-DCMAKE_CXX_FLAGS="-g" \
-DCMAKE_C_FLAGS="-g" \
-DSSIMLIB=/lus/eagle/projects/datascience/balin/SDL_Workshop/SmartRedis \
./

make
