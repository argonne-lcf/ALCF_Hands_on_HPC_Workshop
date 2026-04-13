#!/bin/bash
#PBS -A gpu_hack
#PBS -l walltime=10:00
#PBS -l filesystems=flare
#PBS -q debug

# The rest is an example of how an MPI job might be set up
echo Working directory is `pwd`

echo Jobid: $PBS_JOBID
echo Running on host `hostname`
echo Running on nodes `cat $PBS_NODEFILE`

module load hdf5/1.14.3
module list

export MPIR_CVAR_CH4_XPMEM_ENABLE=0
export MPIR_CVAR_ENABLE_GPU=0
unset MPIR_CVAR_CH4_COLL_SELECTION_TUNING_JSON_FILE
unset MPIR_CVAR_COLL_SELECTION_TUNING_JSON_FILE
unset MPIR_CVAR_CH4_POSIX_COLL_SELECTION_TUNING_JSON_FILE
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_CQ_FILL_PERCENT=20

export LIBOMP_USE_HIDDEN_HELPER_TASK=0
export ZES_ENABLE_SYSMAN=1
export SYCL_JIT_CACHE_SIZE=0

export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
export NEOReadDebugKeys=1
export EnableRecoverablePageFaults=0
export SplitBcsCopy=0

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS=12         # Number of MPI ranks per node
NDEPTH=8          # Number of hardware threads per rank, spacing between MPI ranks on a node

NRANKS=6
NTOTRANKS=$(( NNODES * NRANKS ))
echo "NUM_NODES=${NNODES}  TOTAL_RANKS=${NTOTRANKS}  RANKS_PER_NODE=${NRANKS}  THREADS_PER_RANK=${OMP_NUM_THREADS}"

echo # blank line
echo Information about the i915 GPU driver:
cat /sys/module/i915/version

exe_root=/flare/alcf_training/hpctoolkit_examples/.qmcpack
exe_root=/home/johnmc/work/hackathon
exe_bin=$exe_root/qmcpack/build_aurora_oneapi2025.3.1_gpu_real_MP/bin

echo # blank line
echo "Output from ldd qmcpack (showing all of the libraries linked with it):"
ldd $exe_bin/qmcpack

export OMP_NUM_THREADS=8
export ExperimentalH2DCpuCopyThreshold=50000

CPU_BIND=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100
CPU_BIND_VERBOSE=verbose,$CPU_BIND

module use /soft/perftools/hpctoolkit/modulefiles
module load hpctoolkit

HPCRUN_BIN=`which hpcrun`

export HPCRUN="$HPCRUN_BIN -e CPUTIME -tt -e gpu=level0 -e gpu=opencl -o qmcpack.m"

COMPACT=/soft/tools/mpi_wrapper_utils/gpu_tile_compact.sh 

QMCPACK=$exe_bin/qmcpack 
QMCPACK_INPUT=/home/johnmc/work/inputs/NiO-fcc-S128-dmc.xml

echo # blank line
echo "Start time: `date`"
echo # blank line
echo Executing the following command:
cmd="mpiexec --envall -np ${NTOTRANKS} -ppn ${NRANKS} -d ${NDEPTH} --cpu-bind $CPU_BIND $COMPACT $HPCRUN $QMCPACK --enable-timers=fine $QMCPACK_INPUT "
echo $cmd
eval $cmd
rc=$?
echo # blank line
echo "End time: `date`"
exit $?
