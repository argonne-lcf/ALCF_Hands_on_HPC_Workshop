#!/bin/bash --login
# COBALT -n 1
#COBALT -q single-gpu
#COBALT -A datascience
#COBALT --attrs filesystems=home,theta-fs0,grand,eagle
# -------------------------------------------------------
# UG Section 2.5, page UG-24 Job Submission Options
# Add another # at the beginning of the line to comment out a line
# NOTE: adding a switch to the command line will override values in this file.

# These options are MANDATORY at ALCF; Your qsub will fail if you don't provide them.
##PBS -A <short project name>
##PBS -l walltime=HH:MM:SS

# Highly recommended 
# The first 15 characters of the job name are displayed in the qstat output:
##PBS -N <name>

# If you need a queue other than the default (uncomment to use)
##PBS -q <queue name>
# Controlling the output of your application
# UG Sec 3.3 page UG-40 Managing Output and Error Files
# By default, PBS spools your output on the compute node and then uses scp to move it the
# destination directory after the job finishes.  Since we have globally mounted file systems
# it is highly recommended that you use the -k option to write directly to the destination
# the doe stands for direct, output, error
##PBS -o <path for stdout>
##PBS -k doe
##PBS -e <path for stderr>
# Setting job dependencies
# UG Section 6.2, page UG-107 Using Job Dependencies
# There are many options for how to set up dependancies;  afterok will give behavior similar
# to Cobalt (uncomment to use)
##PBS depend=afterok:<jobid>:<jobid>

# Environment variables (uncomment to use)
# Section 6.12, page UG-126 Using Environment Variables
# Sect 2.59.7, page RG-231 Enviornment variables PBS puts in the job environment
##PBS -v <variable list>
## -v a=10, "var2='A,B'", c=20, HOME=/home/zzz
##PBS -V exports all the environment variables in your environnment to the compute node
# The rest is an example of how an MPI job might be set up
#echo Working directory is $PBS_O_WORKDIR
##cd $PBS_O_WORKDIR

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd -LP)
PARENT=$(dirname "${DIR}")
ROOT=$(dirname "${PARENT}")
echo "cwd: $DIR"
echo "parent: $PARENT"
echo "ROOT: $ROOT"
printf '%.s─' $(seq 1 $(tput cols))

HOST=$(hostname)

TSTAMP=$(date "+%Y-%m-%d-%H%M%S")
echo "Job started at: ${TSTAMP}"

export NCPU_PER_RANK=$(getconf _NPROCESSORS_ONLN)

if [[ $(hostname) == x* ]]; then
  # echo "Job ID: ${PBS_JOBID}"
  # echo Working directory is $PBS_O_WORKDIR
  # cd $PBS_O_WORKDIR
  HOSTFILE="${PBS_NODEFILE}"
  export ALCF_RESOURCE="polaris"
  export NRANKS=$(wc -l < "${PBS_NODEFILE}")
  export NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
  export NGPUS="$((${NRANKS}*${NGPU_PER_RANK}))"
  echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
  echo "┃  RUNNING ON ${ALCF_RESOURCE}: ${NGPUS} GPUs   ┃"
  echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
  module load conda; conda activate base
  VENV_PREFIX='2022-09-08'
  MPI_COMMAND=$(which mpiexec)
  MPI_FLAGS="--envall \
    -n ${NGPUS} \
    --depth=${NCPU_PER_RANK} \
    --ppn ${NGPU_PER_RANK} \
    --hostfile ${PBS_NODEFILE}"
elif [[ $(hostname) == theta* ]]; then
  export ALCF_RESOURCE="thetaGPU"
  export HOSTFILE="${COBALT_NODEFILE}"
  export NRANKS=$(wc -l < "${COBALT_NODEFILE}")
  export NGPU_PER_RANK=$(nvidia-smi -L | wc -l)
  export NGPUS=$(("${NRANKS}"*"${NGPU_PER_RANK}"))
  echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓"
  echo "┃  RUNNING ON ${ALCF_RESOURCE}: ${NGPUS} GPUs   ┃"
  echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛"
  module load conda; conda activate base
  VENV_PREFIX='2022-07-21'
  MPI_COMMAND=$(which mpirun)
  MPI_FLAGS="-x LD_LIBRARY_PATH \
    -x PATH \
    --verbose \
    -n ${NGPUS} \
    -npernode ${NGPU_PER_RANK} \
    --hostfile ${HOSTFILE}"
else
  export ALCF_RESOURCE="NONE"
  echo "HOSTNAME: $(hostname)"
fi

# -----------------------------------------------
# - Get number of global CPUs by multiplying: 
#        (# CPU / rank) * (# ranks)
# -----------------------------------------------
export NCPU_PER_RANK=$(getconf _NPROCESSORS_ONLN)
export NCPUS=$(("${NRANKS}"*"${NCPU_PER_RANK}"))


# ---- Specify directories and executable for experiment ------------------
MAIN="${DIR}/main.py"
SET_AFFINITY="${DIR}/affinity.sh"
LOGDIR="${DIR}/logs"
LOGFILE="${LOGDIR}/${TSTAMP}-${HOST}_ngpu${NGPUS}.log"
if [ ! -d "${LOGDIR}" ]; then
  mkdir -p ${LOGDIR}
fi

# Keep track of latest logfile for easy access
echo $LOGFILE >> "${DIR}/logs/latest"

# Double check everythings in the right spot
echo "DIR=${DIR}"
# echo "EXEC=${EXEC}"
echo "PARENT=${PARENT}"
echo "ROOT=${ROOT}"
echo "LOGDIR=${LOGDIR}"
echo "LOGFILE=${LOGFILE}"

conda run python3 -m pip install --upgrade pip

# -----------------------------------------------------------
# 1. Check if a virtual environment exists in project root: 
#    `sdl_workshop/hyperparameterManagement/`
#
# 2. If so, activate environment and make sure we have an 
#    editable install
# -----------------------------------------------------------
VENV_DIR="${ROOT}/venvs/${ALCF_RESOURCE}/${VENV_PREFIX}"
if [ -d ${VENV_DIR} ]; then
  echo "Found venv at: ${VENV_DIR}"
  source "${VENV_DIR}/bin/activate"
else
  echo "Creating new venv at: ${VENV_DIR}"
  python3 -m venv ${VENV_DIR} --system-site-packages
  source "${VENV_DIR}/bin/activate"
fi

# ---- Environment settings ------------------------------------------
# export NCCL_DEBUG=INFO
# export KMP_SETTINGS=TRUE
# export OMP_NUM_THREADS=16
# export TF_ENABLE_AUTO_MIXED_PRECISION=1
chmod +x ${SET_AFFINITY}


EXEC="${MPI_COMMAND} ${MPI_FLAGS} ${SET_AFFINITY} $(which python3) ${MAIN}"

printf '%.s─' $(seq 1 $(tput cols))

# ------ Print job information --------------------------------------+
echo "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "┃  STARTING A NEW RUN ON ${NGPUS} GPUs of ${ALCF_RESOURCE}"
echo "┃━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "┃  - Writing logs to: ${LOGFILE}"
echo "┃  - DATE: ${TSTAMP}"
echo "┃  - NRANKS: $NRANKS"
echo "┃  - NGPUS PER RANK: ${NGPU_PER_RANK}"
echo "┃  - NGPUS TOTAL: ${NGPUS}"
echo "┃  - python3: $(which python3)"
echo "┃  - MPI: ${MPI_COMMAND}"
echo "┃  - exec: ${EXEC}"
echo "┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "Latest logfile: $(tail -1 ./logs/latest)"

${EXEC} $@ > ${LOGFILE}
