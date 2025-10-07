#! /bin/bash -x
#
# Time Stamp
tstamp() {
     date +"%Y-%m-%d-%H%M%S"
}

## Load modules 2025.1.3 with runtime (UMD) 1099.17
#
#module restore
#module unload mpich oneapi
#module use /soft/compilers/oneapi/nope/modulefiles
#module use /soft/compilers/oneapi/2025.1.3/modulefiles
#module use /soft/preview/components/graphics-compute-runtime/1099.17/modulefiles
#module add mpich/nope/develop-git.6037a7a
#module add oneapi/public/2025.1.3
#module add graphics-compute-runtime/1099.17

#module load cmake
#unset CMAKE_ROOT

#source /opt/aurora/24.347.0/spack/unified/0.9.2/install/linux-sles15-x86_64/gcc-13.3.0/miniforge3-24.3.0-0-gfganax/bin/activate
#conda activate /lus/flare/projects/datasets/softwares/envs/vLLM_main_pytorch_2p8_oneapi_2025p1p3_pti_0p10p3_python3p10_julia
#source /lus/flare/projects/datasets/softwares/envs/pytorch_2p8_oneapi_2025p1p3_umd_1099p17_mpi4py/bin/activate

source /lus/flare/projects/datasets/softwares/training/hpc_hands_10_07_2025/for_workshop/hpc_hands_on_2025_pytorch_2p8.env

NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=2

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

N=2
PPN=2
EPOCHS=10

NODES=1

TRACE_DIR_ROOT=/lus/flare/projects/datasets/softwares/training/hpc_hands_10_07_2025/for_workshop/traces/pytorch_2p8
TRACE_DIR=${TRACE_DIR_ROOT}/xpu_compile_pt_2p8_E${EPOCHS}_N${NODES}_R${PPN}_$(tstamp)

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export FI_MR_CACHE_MONITOR=userfaultfd

export CCL_PROCESS_LAUNCHER=pmix

#export CPU_AFFINITY="list:4-7:8-11:12-15:16-19:20-23:24-27:56-59:60-63:64-67:68-71:72-75:76-79"
#export CCL_WORKER_AFFINITY="42,43,44,45,46,47,94,95,96,97,98,99"
#export ZE_AFFINITY_MASK="0,1,2,3,4,5,6,7,8,9,10,11"

export CPU_AFFINITY="list:4-7:8-11"
export CCL_WORKER_AFFINITY="42,43"
export ZE_AFFINITY_MASK="0,1"


#export CPU_AFFINITY="list:4-7"
#export CCL_WORKER_AFFINITY="42"
#export ZE_AFFINITY_MASK="0"

mpiexec -n ${N} -ppn ${PPN} -l --line-buffer --cpu-bind ${CPU_AFFINITY} \
    python pytorch_2p8_ddp_compile_prof.py --epochs ${EPOCHS} --trace-dir ${TRACE_DIR} 
