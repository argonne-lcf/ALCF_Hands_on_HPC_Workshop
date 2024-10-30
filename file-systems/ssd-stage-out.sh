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

# run the benchmark outputting to Lustre scratch storage
echo "***********************************"
echo "*********** LUSTRE TEST ***********"
echo "***********************************"
mpiexec -n 64 ./mpi-io-test -w -f $LUSTRE_SCRATCH/test.out
rm $LUSTRE_SCRATCH/test.out
echo ""

# run the benchmark outputting to node-local SSD storage
echo "**********************************"
echo "************ SSD TEST ************"
echo "**********************************"
mpiexec -n 64 ./mpi-io-test -w -f /local/scratch/test.out
echo ""

# stage out the output file to Lustre scratch for long-term storage
mv /local/scratch/test.out $LUSTRE_SCRATCH/mpi-io-test.stageout
ls -alh $LUSTRE_SCRATCH
