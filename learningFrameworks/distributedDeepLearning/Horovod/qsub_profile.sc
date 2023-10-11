#!/bin/bash -l
#PBS -l walltime=00:30:00
#PBS -l nodes=2:ppn=4
#PBS -N cnn_horovod
#PBS -k doe
#PBS -j oe
#PBS -A fallwkshp23
#PBS -q fallws23scaling
#PBS -l filesystems=home:eagle:grand

cd $PBS_O_WORKDIR

module load conda/2023-10-04; conda activate
LD_PRELOAD=/soft/perftools/mpitrace/lib/libmpitrace.so aprun -n 8 -N 4 --cc depth -d 16 python keras_cnn_concise_hvd.py --epochs 10
HOROVOD_TIMELINE=horovod_timeline.json aprun -n 8 -N 4 --cc depth -d 16 python keras_cnn_concise_hvd.py --epochs 10
