#!/bin/bash -l
#PBS -l select=1
#PBS -l walltime=00:20:00
#PBS -l filesystems=eagle:home_fs
#PBS -q debug
#PBS -o logdir/
#PBS -e logdir/

cd $PBS_O_WORKDIR

echo [$SECONDS] setup conda environment
module load conda/2022-09-08
conda activate

echo [$SECONDS] python = $(which python)
echo [$SECONDS] python version = $(python --version)

echo [$SECONDS] setup local env vars
NODES=`cat $PBS_NODEFILE | wc -l`
GPUS_PER_NODE=4
RANKS=$((NODES * GPUS_PER_NODE))
echo NODES=$NODES  GPUS_PER_NODE=$GPUS_PER_NODE  RANKS=$RANKS

export OMP_NUM_THREADS=1
echo [$SECONDS] run example with $OMP_NUM_THREADS threads
mpiexec -n $RANKS --ppn $GPUS_PER_NODE --depth=$OMP_NUM_THREADS --cpu-bind depth --env OMP_NUM_THREADS=$OMP_NUM_THREADS -env OMP_PLACES=threads \
   python ilsvrc_dataset.py -c ilsvrc.json --interop $OMP_NUM_THREADS --intraop $OMP_NUM_THREADS \
       --logdir logdir/t${OMP_NUM_THREADS}_${PBS_JOBID}

export OMP_NUM_THREADS=16
echo [$SECONDS] run example with $OMP_NUM_THREADS threads
mpiexec -n $RANKS --ppn $GPUS_PER_NODE --depth=$OMP_NUM_THREADS --cpu-bind depth --env OMP_NUM_THREADS=$OMP_NUM_THREADS -env OMP_PLACES=threads \
   python ilsvrc_dataset.py -c ilsvrc.json --interop $OMP_NUM_THREADS --intraop $OMP_NUM_THREADS \
       --logdir logdir/t${OMP_NUM_THREADS}_${PBS_JOBID}

export OMP_NUM_THREADS=64
echo [$SECONDS] run example with $OMP_NUM_THREADS threads
mpiexec -n $RANKS --ppn $GPUS_PER_NODE --depth=$OMP_NUM_THREADS --cpu-bind depth --env OMP_NUM_THREADS=$OMP_NUM_THREADS -env OMP_PLACES=threads \
   python ilsvrc_dataset.py -c ilsvrc.json --interop $OMP_NUM_THREADS --intraop $OMP_NUM_THREADS \
       --logdir logdir/t${OMP_NUM_THREADS}_${PBS_JOBID}

echo [$SECONDS] done
