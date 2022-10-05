#!/bin/bash

# change `CONDA_ENV_PREFIX` with the path to your conda environment
CONDA_ENV=/lus/eagle/projects/SDL_Workshop/SmartSim/ssim
DRIVER=src/driver.py

echo database nodes $1
echo simulation nodes $2
echo ml nodes $3
nodes=$(($1 + $2 + $3))
echo total nodes $nodes
echo CPU cores per node $4
echo simprocs $5
echo sim_ppn $6
echo mlprocs $7
echo ml_ppn $8
echo device $9
echo verbose ${10}

module load conda/2022-09-08
conda activate $CONDA_ENV
HOST_FILE=$(echo $PBS_NODEFILE)

echo 2 > input.config
echo "$1 $4" >> input.config

python $DRIVER $nodes $1 $2 $3 $4 $5 $6 $7 $8 $9 ${10} $HOST_FILE
