#!/bin/bash

# change `CONDA_ENV_PREFIX` with the path to your conda environment
CONDA_ENV=/lus/grand/projects/datascience/balin/Polaris/smartsim_envs/buildFromClean_preProd/ssim
DRIVER=src/driver.py

echo nodes $1
echo CPU cores per node $2
echo simprocs $3
echo sim_ppn $4
echo mlprocs $5
echo ml_ppn $6
echo db_ppn $7
echo device $8
echo logging $9

module load conda/2022-07-19
conda activate $CONDA_ENV
HOST_FILE=$(echo $PBS_NODEFILE)

python $DRIVER $1 $2 $3 $4 $5 $6 $7 $8 $9 $HOST_FILE
