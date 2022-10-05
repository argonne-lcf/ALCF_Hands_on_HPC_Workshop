# Kokkos

### Build and Install Kokkos

```
git clone https://github.com/kokkos/kokkos.git
cd kokkos
# Edit default_arch in ./bin/nvcc_wrapper to sm_80
mkdir build
cd build

export KOKKOS_SRC=

cmake -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=$KOKKOS_SRC/bin/nvcc_wrapper \
  -DCMAKE_CXX_EXTENSIONS=Off -DCMAKE_CXX_STANDARD=14 \
  -DCMAKE_INSTALL_PREFIX=$KOKKOS_SRC/install \
  -DBUILD_STATIC_LIBS=On -DBUILD_SHARED_LIBS=On \
  -DKokkos_ENABLE_CUDA=On -DKokkos_ENABLE_CUDA_LAMBDA=On \
  -DKokkos_ENABLE_OPENMP=On -DKokkos_ENABLE_SERIAL=On \
  -DKokkos_ENABLE_TESTS=On \
  ..

make -j 18
make install
```

### Build and Run Kokkos Benchmark
```
cd kokkos/benchmarks/stream
# Edit Makefile to use KOKKOS_ARCH="Ampere80"
make -j 8
./stream-cuda
```
