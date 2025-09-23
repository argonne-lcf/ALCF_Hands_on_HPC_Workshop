#!/bin/bash -x
#PBS -l select=1
#PBS -l walltime=00:20:00
#PBS -A datascience
#PBS -q debug-scaling
#PBS -k doe
#PBS -l filesystems=flare

export HTTP_PROXY=http://proxy.alcf.anl.gov:3128
export HTTPS_PROXY=http://proxy.alcf.anl.gov:3128
export http_proxy=http://proxy.alcf.anl.gov:3128
export https_proxy=http://proxy.alcf.anl.gov:3128

# This example shows loading python modules from a lustre directory with using copper.

# create a conda environment and install numpy
LUS_CONDA_PATH=${HOME}/copper_test_env
source $IDPROOT/etc/profile.d/conda.sh
conda create -y -p ${LUS_CONDA_PATH} numpy
 
cd $PBS_O_WORKDIR
echo Jobid: $PBS_JOBID
echo Running on nodes `cat $PBS_NODEFILE`

NNODES=`wc -l < $PBS_NODEFILE`
RANKS_PER_NODE=12
NRANKS=$(( NNODES * RANKS_PER_NODE ))
echo "App running on NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NRANKS}  RANKS_PER_NODE=${RANKS_PER_NODE}"


# import without copper 
conda activate ${LUS_CONDA_PATH}
echo "import without copper"
mpirun --np ${NRANKS} --ppn ${RANKS_PER_NODE} \
    --cpu-bind=list:4:9:14:19:20:25:56:61:66:71:74:79 --genvall \
    python3 -c "import numpy; print(numpy.__file__)"
conda deactivate

module load copper
launch_copper.sh
# Prepend /tmp/${USER}/copper/ to all your absolute paths if you want your I/O to go through copper (including PYTHON_PATH, CONDA_PREFIX, CONDA_ROOT and PATH)

# import with copper
IMAGE_VERSION="24.347.0"
ONEAPI_VERSION="2025.0"
export LD_LIBRARY_PATH=/opt/aurora/${IMAGE_VERSION}/oneapi/${ONEAPI_VERSION}/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/aurora/${IMAGE_VERSION}/oneapi/compiler/${ONEAPI_VERSION}/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/aurora/${IMAGE_VERSION}/oneapi/intel-conda-miniforge/envs/${ONEAPI_VERSION}.0/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/aurora/${IMAGE_VERSION}/oneapi/intel-conda-miniforge/pkgs/intel-sycl-rt-2025.0.4-intel_1519/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/aurora/${IMAGE_VERSION}/updates/oneapi/compiler/${ONEAPI_VERSION}/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/aurora/${IMAGE_VERSION}/support/tools/pti-gpu/0.11.0/lib64/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${LUS_CONDA_PATH}:$LD_LIBRARY_PATH

#export CONDA_PREFIX=/tmp/${USER}/copper/${LUS_CONDA_PATH}/bin:$CONDA_PREFIX
export PATH=/tmp/${USER}/copper/${LUS_CONDA_PATH}/bin:$PATH
export PYTHONPATH=/tmp/${USER}/copper/${LUS_CONDA_PATH}:$PYTHONPATH
echo "import with copper"
mpirun --np ${NRANKS} --ppn ${RANKS_PER_NODE} \
    --cpu-bind=list:4:9:14:19:20:25:56:61:66:71:74:79 --genvall \
    python3 -c "import numpy; print(numpy.__file__)"

# stop copper (optional)
stop_copper.sh

# delete the conda environment
conda remove -p ${LUS_CONDA_PATH} -y --all
