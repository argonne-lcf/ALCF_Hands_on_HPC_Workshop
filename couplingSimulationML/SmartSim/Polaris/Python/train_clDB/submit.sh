#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N train_clDB
#PBS -l walltime=00:30:00
#PBS -l select=3:ncpus=64:ngpus=4
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
mlnodes=1
nodes=$(($dbnodes + $simnodes + $mlnodes))
ppn=64 # CPU cores per node
simprocs=64
sim_ppn=64 # CPU cores per node assigned to sim
mlprocs=4
ml_ppn=4 # ranks per node assigned to ML
device=cuda
logging="no"

echo number of database nodes $dbnodes
echo number of simulation nodes $simnodes
echo number of ML nodes $mlnodes
echo number of total nodes $nodes
echo number of sim processes $simprocs
echo number of sim processes per node $sim_ppn
echo number of ML processes $mlprocs
echo number of ML processes per node $ml_ppn
echo device for ML $device
echo CPU cores per node $ppn
echo conda environment $CONDA_ENV

# Set env
cd $PBS_O_WORKDIR
module load $MODULE
conda activate $CONDA_ENV
HOST_FILE=$(echo $PBS_NODEFILE)

# Run
echo python $DRIVER $nodes $dbnodes $simnodes $mlnodes $ppn $simprocs $sim_ppn $mlprocs $ml_ppn $device $logging $HOST_FILE
python $DRIVER $nodes $dbnodes $simnodes $mlnodes $ppn $simprocs $sim_ppn $mlprocs $ml_ppn $device $logging $HOST_FILE

# Handle output
if [ "$logging" = "verbose" ]; then
    mkdir $PBS_JOBID
    mv *.log $PBS_JOBID
fi
