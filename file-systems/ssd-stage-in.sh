#!/bin/bash
#PBS -l select=1:system=polaris
#PBS -l filesystems=home:grand
#PBS -l walltime=0:15:00
#PBS -A alcf_training
#PBS -q HandsOnHPC

cd ${PBS_O_WORKDIR}

# setup user-specific scratch in project directory on Grand
LUSTRE_SCRATCH=/grand/alcf_training/$USER/file-systems
mkdir -p $LUSTRE_SCRATCH

# NOTE: this job script assumes you have already ran the 
#       ssd-stage-out example job

# run the benchmark reading from Lustre scratch storage
echo "***********************************"
echo "*********** LUSTRE TEST ***********"
echo "***********************************"
mpiexec -n 64 ./mpi-io-test -r -f $LUSTRE_SCRATCH/mpi-io-test.stageout
echo ""

# stage in the benchmark input file from Lustre scratch
cp $LUSTRE_SCRATCH/mpi-io-test.stageout /local/scratch
ls -alh /local/scratch
echo ""

# run the benchmark reading from node-local SSD storage
echo "**********************************"
echo "************ SSD TEST ************"
echo "**********************************"
mpiexec -n 64 ./mpi-io-test -r -f /local/scratch/mpi-io-test.stageout
echo ""
