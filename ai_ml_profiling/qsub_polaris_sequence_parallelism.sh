#!/bin/bash -l
#PBS -l select=1:system=polaris
#PBS -l place=scatter
#PBS -l walltime=00:05:00
#PBS -q debug
#PBS -l filesystems=home:eagle
#PBS -A datascience
#PBS -N NSYS_1x2
#PBS -k doe
#PBS -o /home/hossainm/hpc_workshop_october_2024/outdir
#PBS -e /home/hossainm/hpc_workshop_october_2024/errordir
#PBS -j oe


# What's the benchmark work directory?
WORK_DIR=/home/hossainm/hpc_workshop_october_2024
LOG_WRAPPER=${WORK_DIR}/log_wrapper.sh
#cd ${WORK_DIR}

TRIAL=3

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

module reset
module use /soft/modulefiles/
module load conda/2024-04-29
conda activate

echo "====== ENVIRONMENT: MODULE LIST ======"
module list
echo "====== ENVIRONMENT: MODULE LIST ======"

echo "====== ENVIRONMENT: PRINTENV ======"
env
echo "====== ENVIRONMENT: PRINTENV ======"

echo "====== ENVIRONMENT: NCCL ======"
printenv | grep "CCL"
echo "====== ENVIRONMENT: NCCL ======"


RUN_ID=profiling_seq_parallel_fp32-ranks${NRANKS}-nodes${NNODES}-T${TRIAL}

mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} --cpu-bind=verbose,list:4:24:36:48 -l --line-buffer \
python ${WORK_DIR}/sequence_parallelism_compute.py -s 4608 -d 9216 -p "float32"    
