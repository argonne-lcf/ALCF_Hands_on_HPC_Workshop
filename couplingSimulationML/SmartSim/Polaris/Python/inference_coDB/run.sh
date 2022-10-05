#!/bin/bash

# change `CONDA_ENV_PREFIX` with the path to your conda environment
CONDA_ENV=/lus/eagle/projects/SDL_Workshop/SmartSim/ssim
DRIVER=src/driver.py

echo nodes $1
echo CPU cores per node $2
echo simprocs $3
echo sim_ppn $4
echo db_ppn $5
echo device $6
echo logging $7

module load conda/2022-09-08
conda activate $CONDA_ENV
HOST_FILE=$(echo $PBS_NODEFILE)

python $DRIVER $1 $2 $3 $4 $5 $6 $7 $HOST_FILE
