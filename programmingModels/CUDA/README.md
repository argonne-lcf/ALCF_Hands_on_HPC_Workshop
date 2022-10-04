# CUDA

`nvcc --version`

### Build and Run BabelStream

```
git clone https://github.com/UoB-HPC/BabelStream.git
cd BabelStream
module load cmake
cmake -Bbuild -H. -DMODEL=cuda -DCMAKE_CUDA_COMPILER=nvcc -DCUDA_ARCH=sm_80
cd build
make
./cuda-stream -s 102400000
```

### Build and Run DGEMM/SGEMM

```
git clone https://github.com/ParRes/Kernels.git 
cd Kernels/Cxx11
nvcc -g -O3 -std=c++11 -arch=sm_80 -D_MWAITXINTRIN_H_INCLUDED -DPRKVERSION="2020" -DPRK_USE_CUBLAS dgemm-cublas.cu -lcublas -lcublasLt -o dgemm-cublas
./dgemm-cublas 10 20000 0 0 
```
