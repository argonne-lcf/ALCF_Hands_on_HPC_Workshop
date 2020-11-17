#!/bin/sh

# Clean up the directory
rm -rf *.png
cd app_build/

rm -rf *.png
rm -rf *.npy
rm -rf checkpoints/
rm -rf __pycache__/
rm -rf CMakeFiles/
rm -rf app
rm -rf cmake_install.cmake
rm -rf Makefile
rm -rf CMakeCache.txt

cmake ../
make
