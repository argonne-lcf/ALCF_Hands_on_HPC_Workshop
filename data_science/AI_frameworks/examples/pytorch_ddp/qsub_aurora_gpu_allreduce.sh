#!/bin/bash -x
#PBS -l select=4
#PBS -l place=scatter
#PBS -l walltime=00:20:00
#PBS -q gpu_hack_prio
#PBS -A gpu_hack
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -e /home/${USER}
#PBS -o /home/${USER}
#PBS -j oe
#PBS -N ARDC_PT 

## Timezone US/Central
export TZ='/usr/share/zoneinfo/US/Central'

# Define a timestamp function
timestamp() {
  date +"%Y-%m-%d %H:%M:%S" # current time
}

echo "$(timestamp): Start of the Run, after exporting TZ Central"

WORK_DIR=/home/${USER}/GettingStarted/AI_frameworks/examples/pytorch_ddp
LOG_WRAPPER=${WORK_DIR}/log_wrapper.sh 

TRIAL=1
MSG=268435456 ## 1.07 GB per rank, FP32

PRECISION="float32"

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=12

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

module load frameworks

export CCL_KVS_MODE=mpi
export CCL_KVS_CONNECTION_TIMEOUT=300
export FI_MR_CACHE_MONITOR=userfaultfd

## For TP=12, PPN=12
# Using the affinity from the user-guide 
export CPU_AFFINITY="list:4-7:8-11:12-15:16-19:20-23:24-27:56-59:60-63:64-67:68-71:72-75:76-79"
export CCL_WORKER_AFFINITY="42,43,44,45,46,47,94,95,96,97,98,99"
export HOROVOD_THREAD_AFFINITY="7,11,15,19,23,27,59,63,67,71,75,79"
export MEM_BIND="list:2:2:2:2:2:2:3:3:3:3:3:3"
export ZE_AFFINITY_MASK="0,1,2,3,4,5,6,7,8,9,10,11"


echo "========= ENVIRONMENT VARIABLES ======="
env
echo "========= ENVIRONMENT VARIABLES ======="

echo ""

echo "========= CCL VARIABLES =============="
printenv | grep "CCL"
echo "========= CCL VARIABLES =============="


RUN_ID=aurora_ALLREDUCE_1GB_${PRECISION}_N${NNODES}_R${NRANKS_PER_NODE}_T${TRIAL}_$(date +"%Y-%m-%d_%H-%M-%S")

echo "${RUN_ID}"

echo "$(timestamp): Before mpiexec."

mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} -l --line-buffer --cpu-bind ${CPU_AFFINITY} --mem-bind ${MEM_BIND} \
    ${LOG_WRAPPER} python ${WORK_DIR}/gpu_allreduce.py --tensor_dimension_1d=${MSG} --precision=${PRECISION}

echo "$(timestamp): Finished the workload."

