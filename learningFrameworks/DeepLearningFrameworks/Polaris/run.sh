#!/bin/bash
#PBS -l walltime=00:10:00
#PBS -A alcf_training
#PBS -q HandsOnHPC
#PBS -l select=2
#PBS -l filesystems=home,eagle

module load frameworks
python3 run.py