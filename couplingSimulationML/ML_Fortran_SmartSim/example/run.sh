#!/bin/bash

# change `CONDA_ENV_PREFIX` with the path to your conda environment
CONDA_ENV_PREFIX=/projects/SDL_Workshop/sdl_ai_workshop/05_Simulation_ML/ML_Fortran_SmartSim/ssim
DRIVER=src/driver.py

module swap PrgEnv-intel PrgEnv-gnu
export CRAYPE_LINK_TYPE=dynamic

echo ppn $1
echo nodes $2
echo allprocs $3
echo dbnodes $4
echo simnodes $5
echo mlnodes $6
echo simprocs $7
echo mlprocs $8

module load miniconda-3/2021-07-28
conda activate $CONDA_ENV_PREFIX

python $DRIVER $1 $2 $3 $4 $5 $6 $7 $8
