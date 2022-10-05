#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N inf_clDB
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

dbnodes=1
simnodes=1
nodes=$(($dbnodes + $simnodes))
ppn=64 # CPU cores per node
simprocs=64
sim_ppn=64 # CPU cores per node assigned to sim
device=gpu
logging="no"

echo number of total nodes $nodes
echo number of database nodes $dbnodes
echo number of simulation nodes $simnodes
echo number of sim processes $simprocs
echo number of sim processes per node $sim_ppn
echo CPU cores per node $ppn
echo conda environment $CONDA_ENV

# Set env
cd $PBS_O_WORKDIR
module load $MODULE
conda activate $CONDA_ENV
HOST_FILE=$(echo $PBS_NODEFILE)

# Run
echo python $DRIVER $nodes $dbnodes $simnodes $ppn $simprocs $sim_ppn $device $logging $HOST_FILE
echo 2 > input.config
echo "$dbnodes $sim_ppn" >> input.config
python $DRIVER $nodes $dbnodes $simnodes $ppn $simprocs $sim_ppn $device $logging $HOST_FILE

# Handle output
if [ "$logging" = "verbose" ]; then
    mkdir $PBS_JOBID
    mv *.log $PBS_JOBID
fi
