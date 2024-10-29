#!/bin/bash
#PBS -l walltime=00:10:00
#PBS -A alcf_training
#PBS -q HandsOnHPC
#PBS -l select=2
#PBS -l filesystems=home:eagle

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29401


HOSTFILE="${HOSTFILE:-${PBS_NODEFILE}}"
NHOSTS=$(wc -l < "${HOSTFILE}")
export NGPU_PER_HOST=4
export WORLD_SIZE="${WORLD_SIZE:-$(( NHOSTS * NGPU_PER_HOST ))}"

mpiexec --verbose --envall -n $WORLD_SIZE -ppn $NGPU_PER_HOST --genvall --cpu-bind depth -d 16 python3 init_distributed.py