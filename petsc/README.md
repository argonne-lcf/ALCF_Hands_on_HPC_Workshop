# Building

## Get Interactive Node
```
qsub -I -l select=1,walltime=0:60:00,filesystems=home:grand -A alcf_training -q HandsOnHPC
```


## Setup PETSc environment and compile examples
```
cd ALCF_Hands_on_HPC_Workshop/petsc
module reset
module use /soft/modulefiles
module unload darshan
module load cudatoolkit-standalone/12.4.1 PrgEnv-gnu cray-libsci nvhpc-mixed craype-accel-nvidia80
export MPICH_GPU_SUPPORT_ENABLED=1
export MPICH_GPU_IPC_ENABLED=0
export PETSC_DIR=/grand/alcf_training/HandsOnHPC24/petsc
```

## Rosenbrock example
```
make multidim_rosenbrock

// different types
./multidim_rosenbrock -tao_monitor
./multidim_rosenbrock -tao_monitor -tao_type bncg
./multidim_rosenbrock -tao_monitor -tao_type bnls

// play with problem size
./multidim_rosenbrock -tao_converged_reason -tao_monitor -n 1000
./multidim_rosenbrock -tao_converged_reason -tao_monitor -tao_type bncg -n 1000
./multidim_rosenbrock -tao_converged_reason -tao_monitor -tao_type bnls -n 1000

// parallel. play with mpi size and problem size
mpiexec -n 4 ./multidim_rosenbrock -tao_converged_reason -tao_monitor -n 1000
mpiexec -n 4 ./multidim_rosenbrock -tao_converged_reason -tao_monitor -tao_type bncg -n 1000
mpiexec -n 4 ./multidim_rosenbrock -tao_converged_reason -tao_monitor -tao_type bnls -n 1000

// bounded with different starting point
./multidim_rosenbrock -tao_converged_reason -tao_monitor -bound -initial 10
./multidim_rosenbrock -tao_converged_reason -tao_monitor -bound -initial -5.3

// equality constraint
./multidim_rosenbrock -tao_converged_reason -tao_monitor -eq
./multidim_rosenbrock -tao_converged_reason -tao_monitor -eq -initial -1

```

## SNES Example
```
make ex19
export PETSC_OPTIONS="-snes_monitor -snes_converged_reason -lidvelocity 100 -da_grid_x 16 -da_grid_y 16 -ksp_converged_reason -log_view :log.txt"

grep Time\ \(sec\): log.txt

// exact vs inexact Newton
mpiexec -n 1 ./ex19 -da_refine 2 -grashof 1e2
mpiexec -n 1 ./ex19 -da_refine 2 -grashof 1e2 -pc_type lu

//ksp tolerance
mpiexec -n 1 ./ex19 -da_refine 2 -grashof 1e2 -ksp_rtol 1e-8
mpiexec -n 1 ./ex19 -da_refine 2 -grashof 1e2 -ksp_rtol 1e-5
mpiexec -n 1 ./ex19 -da_refine 2 -grashof 1e2 -ksp_rtol 1e-3
mpiexec -n 1 ./ex19 -da_refine 2 -grashof 1e2 -ksp_rtol 1e-2
mpiexec -n 1 ./ex19 -da_refine 2 -grashof 1e2 -ksp_rtol 1e-1

// scaling and parallel
mpiexec -n 12 ./ex19 -ksp_type bcgs -grashof 1e2 -da_refine 2
mpiexec -n 12 ./ex19 -ksp_type bcgs -grashof 1e2 -da_refine 3
mpiexec -n 12 ./ex19 -ksp_type bcgs -grashof 1e2 -da_refine 4

// scaling and parallel with multigrid
mpiexec -n 12 ./ex19 -ksp_type bcgs -grashof 1e2 -pc_type mg -da_refine 2
mpiexec -n 12 ./ex19 -ksp_type bcgs -grashof 1e2 -pc_type mg -da_refine 3
mpiexec -n 12 ./ex19 -ksp_type bcgs -grashof 1e2 -pc_type mg -da_refine 4

// increasing nonlinearity
./ex19 -da_refine 2 -grashof 1e2
./ex19 -da_refine 2 -grashof 1e3
./ex19 -da_refine 2 -grashof 1e4
./ex19 -da_refine 2 -grashof 1.3e4
./ex19 -da_refine 2 -grashof 1.3e4 -pc_type mg
./ex19 -da_refine 2 -grashof 1.3373e4 -pc_type mg
./ex19 -da_refine 2 -grashof 1.3373e4 -pc_type lu

// Nonlinear Richardson Preconditioned with Newton
./ex19 -da_refine 2 -grashof 1.3373e4 -snes_type nrichardson -npc_snes_type newtonls -npc_snes_max_it 4 -npc_pc_type mg
./ex19 -da_refine 2 -grashof 1.3373e4 -snes_type nrichardson -npc_snes_type newtonls -npc_snes_max_it 4 -npc_pc_type lu
./ex19 -da_refine 2 -grashof 1.4e4 -snes_type nrichardson -npc_snes_type newtonls -npc_snes_max_it 4 -npc_pc_type lu

// Newton Preconditioned with Nonlinear Richardson
./ex19 -da_refine 2 -grashof 1.4e4 -pc_type mg -npc_snes_type nrichardson -snes_max_it 1000 -npc_snes_max_it 1
./ex19 -da_refine 2 -grashof 1.4e4 -pc_type mg -npc_snes_type nrichardson -snes_max_it 1000 -npc_snes_max_it 3
./ex19 -da_refine 2 -grashof 1.4e4 -pc_type mg -npc_snes_type nrichardson -snes_max_it 1000 -npc_snes_max_it 4
./ex19 -da_refine 2 -grashof 1.4e4 -pc_type mg -npc_snes_type nrichardson -snes_max_it 1000 -npc_snes_max_it 5
./ex19 -da_refine 2 -grashof 1.4e4 -pc_type mg -npc_snes_type nrichardson -snes_max_it 1000 -npc_snes_max_it 6
./ex19 -da_refine 2 -grashof 1.4e4 -pc_type mg -npc_snes_type nrichardson -snes_max_it 1000 -npc_snes_max_it 7

// extreme case
./ex19 -da_refine 2 -grashof 1e6 -pc_type lu -npc_snes_type nrichardson -snes_max_it 1000 -npc_snes_max_it 7

// GPU
mpiexec -n 1 ./ex19 -da_refine 4 -pc_type mg -mg_levels_pc_type jacobi -pc_mg_log -dm_vec_type cuda -dm_mat_type aijcusparse -log_view_gpu_time -log_view :log_mg_gpu_n1.txt

```
