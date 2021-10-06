#!/bin/bash

CC=cc
CXX=CC
FC=ftn

cmake \
-DCMAKE_Fortran_FLAGS="-craype-verbose -g" \
-DCMAKE_CXX_FLAGS="-craype-verbose -g" \
-DCMAKE_C_FLAGS="-craype-verbose -g" \
./

make
