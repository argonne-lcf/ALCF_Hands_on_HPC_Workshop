#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N train_coDB
#PBS -l walltime=00:30:00
#PBS -l select=2:ncpus=64:ngpus=4
#PBS -l filesystems=eagle:home
#PBS -k doe
#PBS -j oe
#PBS -A SDL_Workshop
#PBS -q SDL_Workshop
#PBS -V

DRIVER=src/driver.py
MODULE=conda/2022-09-08
CONDA_ENV=/lus/eagle/projects/SDL_Workshop/SmartSim/ssim

nodes=2
ppn=64 # CPU cores per node
simprocs=32
sim_ppn=16 # CPU cores per node assigned to sim
mlprocs=8
ml_ppn=4 # ranks per node assigned to ML
db_ppn=8 # CPU cores per node assigned to DB
device=cuda
logging="no"

echo number of nodes $nodes
echo number of sim processes $simprocs
echo number of sim processes per node $sim_ppn
echo number of ML processes $mlprocs
echo number of ML processes per node $ml_ppn
echo number of db processes per node $db_ppn
echo CPU cores per node $ppn
echo conda environment $CONDA_ENV

# Set env
cd $PBS_O_WORKDIR
module load $MODULE
conda activate $CONDA_ENV
HOST_FILE=$(echo $PBS_NODEFILE)

# Run
echo python $DRIVER $nodes $ppn $simprocs $sim_ppn $mlprocs $ml_ppn $db_ppn $device $logging $HOST_FILE
python $DRIVER $nodes $ppn $simprocs $sim_ppn $mlprocs $ml_ppn $db_ppn $device $logging $HOST_FILE

# Handle output
if [ "$logging" = "verbose" ]; then
    mkdir $PBS_JOBID
    mv *.log $PBS_JOBID
fi
