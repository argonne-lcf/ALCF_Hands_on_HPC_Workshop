#!/bin/bash
#COBALT -n 1
#COBALT -t 30
#COBALT -q debug-flat-quad
#COBALT -A SDL_Workshop
#COBALT -O logdir/$COBALT_JOBID

echo [$SECONDS] setup conda environment
module load miniconda-3/2020-12

echo [$SECONDS] setup local env vars
export HYPERTHREADS_PER_CORE=1
export CORES_PER_NODE=64
export OMP_NUM_THREADS=$(( $HYPERTHREADS_PER_CORE * $CORES_PER_NODE ))
export RANKS_PER_NODE=1
export NODES_IN_JOB=$COBALT_PARTSIZE
export RANKS=$(( $RANKS_PER_NODE * $NODES_IN_JOB ))

export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=0

aprun -n $RANKS -N $RANKS_PER_NODE --cc none \
   python ilsvrc_dataset.py -c ilsvrc.json --logdir logdir/$COBALT_JOBID --nthreads $OMP_NUM_THREADS 

