#!/bin/bash -l
#PBS -A alcf_training
#PBS -l select=1
#PBS -N hello_parsl
#PBS -l walltime=0:10:00
#PBS -l filesystems=home:eagle:grand
#PBS -k doe
#PBS -q alcf_training

cd $PBS_O_WORKDIR

module unload xalt
source /grand/alcf_training/workflows/_env/bin/activate

python hello_injob_orchestration.py
