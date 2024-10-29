# Tutorial
```
git clone git@github.com:kokkos/kokkos-tutorials.git
cd kokkos-tutorials/Intro-Short
KokkosTutorial_Short.pdf  # This tutorial
```

## Exercises
```
cd ../Exercises
export MYKOKKOS_EXERCISES="${PWD}"
```

# Building

## Building Kokkos on Polaris

```
module use /soft/modulefiles
module load PrgEnv-nvhpc
export CRAYPE_LINK_TYPE=dynamic

git clone git@github.com:kokkos/kokkos.git
cd kokkos
export MYKOKKOS_BUILD="${PWD}"/build-nvhpc
export MYKOKKOS_INSTALL="${MYKOKKOS_BUILD}"/install
export MYKOKKOS_DIR="${MYKOKKOS_INSTALL}"/lib64/cmake/Kokkos

cmake -B "${MYKOKKOS_BUILD}" -DKokkos_ENABLE_CUDA=ON -DKokkos_ARCH_AMPERE80=On -DCMAKE_CXX_COMPILER=CC -D CMAKE_INSTALL_PREFIX="${MYKOKKOS_INSTALL}"
cmake --build "${MYKOKKOS_BUILD}"/ -- install
```

## Building Exercise #1 on Polaris (other exercises are similar)
```
cd "${MYKOKKOS_EXERCISES}"/01/Begin
cmake -B build-nvhpc -DKokkkos_DIR="${MYKOKKOS_DIR}"
cmake --build build-nvhpc
```
