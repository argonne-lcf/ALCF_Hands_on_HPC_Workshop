#!/bin/bash -l
#PBS -l select=1
#PBS -l walltime=00:20:00
#PBS -l filesystems=eagle:home
#PBS -q alcf_training
#PBS -A alcf_training
#PBS -j oe

export TMPDIR=/tmp/

cd $PBS_O_WORKDIR

echo [$SECONDS] setup conda environment

module load conda
conda activate

echo [$SECONDS] python = $(which python)
echo [$SECONDS] python version = $(python --version)

echo [$SECONDS] setup local env vars
NODES=`cat $PBS_NODEFILE | wc -l`
GPUS_PER_NODE=4
RANKS=$((NODES * GPUS_PER_NODE))
echo NODES=$NODES  GPUS_PER_NODE=$GPUS_PER_NODE  RANKS=$RANKS


# for PyTorch DDT setup
export MASTER_ADDR="localhost"
export MASTER_PORT=12399
echo [$SECONDS] MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT

PROFILE='echo "--profile logdir/$PBS_JOBID/${NUM_WORKERS}_${DEPTH}"'
# PROFILE='echo "--profile"'
# PROFILE='echo ""'

NUM_WORKERS=${NUM_WORKERS:-1}
DEPTH=16
echo [$SECONDS] run example with $NUM_WORKERS:$DEPTH threads
mpiexec -n $RANKS --ppn $GPUS_PER_NODE --depth=$DEPTH --cpu-bind depth \
   python ilsvrc_dataset.py -c ilsvrc.json --num-workers $NUM_WORKERS \
       --logdir logdir $(eval $PROFILE)  |& tee -a run_$NUM_WORKERS.log

