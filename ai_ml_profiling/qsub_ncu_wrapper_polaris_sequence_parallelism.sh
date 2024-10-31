#!/bin/bash -l
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=0:10:00
#PBS -q debug-scaling
#PBS -l filesystems=home:eagle
#PBS -A datascience
#PBS -N NSYS_1x2
#PBS -k doe
#PBS -o /home/hossainm/hpc_workshop_october_2024/outdir
#PBS -e /home/hossainm/hpc_workshop_october_2024/errordir
#PBS -j oe


# What's the benchmark work directory?
WORK_DIR=/home/hossainm/hpc_workshop_october_2024
PROF_DIR=/home/hossainm/hpc_workshop_october_2024/profiles
TEMPORARY_DIR=/eagle/datascience/hossainm/nsys_profile/tmpdir/
LOG_WRAPPER=${WORK_DIR}/log_wrapper.sh
NCU_WRAPPER=${WORK_DIR}/ncu_wrapper.sh 
#cd ${WORK_DIR}

TRIAL=1

# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

module use /soft/modulefiles/
module load conda/2024-04-29
conda activate

#RUN_ID=profiling_seq_parallel_fp32-ranks${NRANKS}-nodes${NNODES}-T${TRIAL}

mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} --env TMPDIR=${TEMPORARY_DIR} --cpu-bind=verbose,list:4:24:36:48 -l --line-buffer \
${NCU_WRAPPER} python ${WORK_DIR}/sequence_parallelism_compute.py --precision "bfloat16"  
