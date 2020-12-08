#!/bin/bash
#COBALT -A SDL_Workshop
#COBALT -q training
#COBALT -t 10
#COBALT -n 1
#COBALT -O logdir/$COBALT_JOBID

echo [$SECONDS] setup conda environment
MCONDA=/lus/theta-fs0/software/thetagpu/conda/tf_master/2020-11-11/mconda3
source $MCONDA/setup.sh

echo [$SECONDS] setup local env vars
NODES=`cat $COBALT_NODEFILE | wc -l`
PPN=1
RANKS=$((NODES * PPN))
echo NODES=$NODES  PPN=$PPN  RANKS=$RANKS

export OMP_NUM_THREADS=64

echo [$SECONDS] run example
python ilsvrc_dataset.py -c ilsvrc.json --interop $OMP_NUM_THREADS --intraop $OMP_NUM_THREADS \
   --logdir logdir/$COBALT_JOBID $HOROVOD 
