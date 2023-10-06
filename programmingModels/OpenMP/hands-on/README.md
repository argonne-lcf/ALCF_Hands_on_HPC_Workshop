# GAMESS RI-MP2 miniapp in Fortran Example

## Setup

1. Get the examples
```
$ git clone https://github.com/jkwack/GAMESS_RI-MP2_MiniApp.git --branch ECP2023
```

2. Submit an interactive job
```
$ qsub -l select=1:system=polaris -l walltime=0:30:00 -l filesystems=home -q fallws23single -A fallwkshp23 -I
```


## Build the CPU version with Cray’s LibSci on a compute node
```
$ cd GAMESS_RI-MP2_MiniApp/
$ make -f Makefile_polaris clean
$ source source_me_ALCF_POLARIS
$ module switch PrgEnv-nvhpc PrgEnv-cray
$ module unload craype-accel-nvidia80
$ make -f Makefile_polaris my_rimp2_cpu
```

## Run it with 32 threads on 1 AMD EYPC Milan with Cray’s LibSci
```
$ OMP_PROC_BIND=spread OMP_NUM_THREADS=32 ./my_rimp2_cpu w30
```

## Build the CPU version with NVBLAS and LibSci on a compute node
```
$ module restore
$ make -f Makefile_polaris my_rimp2
```

## Run it with a A100 GPU with NVBLAS
```
$ CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ./my_rimp2 w30
```

## Build the updated GPU version with NVBLAS and LibSci on a compute node
```
$ make -f Makefile_polaris my_rimp2_v2
```

## Run the update version on a A100 GPU with NVBLAS
```
$ CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 ./my_rimp2_v2 w30
```
