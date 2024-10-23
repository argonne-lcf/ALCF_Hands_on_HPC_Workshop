#!/bin/bash -l
#PBS -l select=1
#PBS -l walltime=00:20:00
#PBS -l filesystems=eagle:home_fs
#PBS -q debug
#PBS -o logdir/
#PBS -e logdir/

cd $PBS_O_WORKDIR

echo [$SECONDS] setup conda environment
module use /soft/modulefiles
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

BATCH_SIZE=64
NSTEPS=100
PROFILE=--profile
# PROFILE=
echo [$SECONDS] using batch size $BATCH_SIZE and $NSTEPS steps

echo [$SECONDS] run serial example
python imagenet_serial.py -b $BATCH_SIZE -s $NSTEPS $PROFILE


NWORKERS=1
echo [$SECONDS] run parallel with $NWORKERS workers
mpiexec -n $RANKS --ppn $GPUS_PER_NODE --depth=16 --cpu-bind depth python imagenet_parallel.py -b $BATCH_SIZE -s $NSTEPS -w $NWORKERS $PROFILE

NWORKERS=2
echo [$SECONDS] run parallel with $NWORKERS workers
mpiexec -n $RANKS --ppn $GPUS_PER_NODE --depth=16 --cpu-bind depth python imagenet_parallel.py -b $BATCH_SIZE -s $NSTEPS -w $NWORKERS $PROFILE

NWORKERS=3
echo [$SECONDS] run parallel with $NWORKERS workers
mpiexec -n $RANKS --ppn $GPUS_PER_NODE --depth=16 --cpu-bind depth python imagenet_parallel.py -b $BATCH_SIZE -s $NSTEPS -w $NWORKERS $PROFILE

NWORKERS=4
echo [$SECONDS] run parallel with $NWORKERS workers
mpiexec -n $RANKS --ppn $GPUS_PER_NODE --depth=16 --cpu-bind depth python imagenet_parallel.py -b $BATCH_SIZE -s $NSTEPS -w $NWORKERS $PROFILE

echo [$SECONDS] done
