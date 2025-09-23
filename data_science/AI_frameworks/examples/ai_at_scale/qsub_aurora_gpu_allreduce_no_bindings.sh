#!/bin/bash -x
#PBS -l select=4
#PBS -l place=scatter
#PBS -l walltime=00:20:00
#PBS -q prod
#PBS -A datascience
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

#ALGO=Ring

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=12

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

module load frameworks

export CCL_KVS_MODE=mpi
export CCL_KVS_CONNECTION_TIMEOUT=300
export FI_MR_CACHE_MONITOR=userfaultfd


echo "========= ENVIRONMENT VARIABLES ======="
env
echo "========= ENVIRONMENT VARIABLES ======="

echo ""

echo "========= CCL VARIABLES =============="
printenv | grep "CCL"
echo "========= CCL VARIABLES =============="


RUN_ID=aurora_NO_BINDINGS_ALLREDUCE_1GB_${PRECISION}_N${NNODES}_R${NRANKS_PER_NODE}_T${TRIAL}_$(date +"%Y-%m-%d_%H-%M-%S")


#LOG_DIR=${WORK_DIR}/run_scripts/outdir/logs 

echo "${RUN_ID}"

echo "$(timestamp): Before mpiexec."

mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} -l --line-buffer \
    ${LOG_WRAPPER} python ${WORK_DIR}/gpu_allreduce.py --tensor_dimension_1d=${MSG} --precision=${PRECISION}

echo "$(timestamp): Finished the workload."

