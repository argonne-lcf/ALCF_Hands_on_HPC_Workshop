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
aprun -n 1 -N 1 --cc depth -d 64 python keras_cnn_concise_hvd.py >& keras_cnn_concise_hvd.py.1.out
aprun -n 2 -N 2 --cc depth -d 32 python keras_cnn_concise_hvd.py >& keras_cnn_concise_hvd.py.2.out
aprun -n 4 -N 4 --cc depth -d 16 python keras_cnn_concise_hvd.py >& keras_cnn_concise_hvd.py.4.out
aprun -n 8 -N 4 --cc depth -d 16 python keras_cnn_concise_hvd.py >& keras_cnn_concise_hvd.py.8.out
