# Tutorial
[KokkosTutorial_Short](https://github.com/kokkos/kokkos-tutorials/blob/main/Intro-Short/KokkosTutorial_Short.pdf)

```
cd

git clone git@github.com:kokkos/kokkos-tutorials.git
cd kokkos-tutorials/Exercises
export MYKOKKOS_EXERCISES="${PWD}"
cd ../Intro-Short    # KokkosTutorial_Short.pdf

```

# Building

## Setup build environment on Polaris

```
cd

module use /soft/modulefiles
module load PrgEnv-nvhpc
export CRAYPE_LINK_TYPE=dynamic

git clone git@github.com:kokkos/kokkos.git
cd kokkos
export MYKOKKOS="${PWD}"

```

## Build OpenMP CPU Backend

```
cmake -B build-openmp -D Kokkos_ENABLE_OPENMP=ON -D Kokkos_ARCH_AMPERE80=ON -D CMAKE_CXX_COMPILER=CC -DCMAKE_INSTALL_PREFIX=build-openmp/install
cmake --build build-openmp/ -- install

```

## Build CUDA GPU Backend

```
cmake -B build-cuda -D Kokkos_ENABLE_CUDA=ON -D Kokkos_ARCH_AMPERE80=ON -D CMAKE_CXX_COMPILER=CC -DCMAKE_INSTALL_PREFIX=build-cuda/install
cmake --build build-cuda/ -- install

```

## Building Exercise #1 on Polaris (other exercises are similar)
```
cd "${MYKOKKOS_EXERCISES}"/01/Begin

cmake -B build-openmp -DKokkkos_DIR="${MYKOKKOS}/build-openmp/install/lib64/cmake/Kokkos
cmake --build build-openmp/

cmake -B build-cuda -DKokkkos_DIR="${MYKOKKOS}/build-cuda/install/lib64/cmake/Kokkos
cmake --build build-cuda/

```
