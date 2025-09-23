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

# setup the 1-stripe directory and run the benchmark
mkdir -p $LUSTRE_SCRATCH/stripe1
# NOTE: no 'lfs setstripe' command needed as 1 stripe is the default
echo "***********************************"
echo "********** 1-STRIPE TEST **********"
echo "***********************************"
mpiexec -n 64 ./mpi-io-test -C -f $LUSTRE_SCRATCH/stripe1/test.out
echo ""
lfs getstripe $LUSTRE_SCRATCH/stripe1/test.out
rm -rf $LUSTRE_SCRATCH/stripe1

# setup the 4-stripe directory and run the benchmark
mkdir -p $LUSTRE_SCRATCH/stripe4
lfs setstripe -c 4 $LUSTRE_SCRATCH/stripe4
echo "***********************************"
echo "********** 4-STRIPE TEST **********"
echo "***********************************"
mpiexec -n 64 ./mpi-io-test -C -f $LUSTRE_SCRATCH/stripe4/test.out
echo ""
lfs getstripe $LUSTRE_SCRATCH/stripe4/test.out
rm -rf $LUSTRE_SCRATCH/stripe4
