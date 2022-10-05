# RAJA

### Build and Install RAJA

```
module load cmake
git clone --recursive https://github.com/LLNL/RAJA.git
cp sdl_workshop/programmingModels/RAJA/scripts/nvcc_gcc.sh RAJA/scripts/
cp sdl_workshop/programmingModels/RAJA/scripts/nvcc_gcc.cmake RAJA/host-configs/alcf-builds/
cd RAJA
./scripts/nvcc_gcc.sh
cd build_alcf_nvcc_gcc
make -j 18
cd bin
./tut_matrix-multiply
```

