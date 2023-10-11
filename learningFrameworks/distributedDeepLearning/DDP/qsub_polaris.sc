#!/bin/bash -l
#PBS -l walltime=00:30:00
#PBS -l nodes=2:ppn=4
#PBS -N cnn_ddp
#PBS -k doe
#PBS -j oe
#PBS -A fallwkshp23
#PBS -l filesystems=home:grand:eagle
#PBS -q fallws23scaling

cd $PBS_O_WORKDIR

module load conda/2023-10-04; conda activate
aprun -n 1 -N 1 --cc depth -d 64 python pytorch_cnn_ddp.py >& pytorch_cnn_ddp.py.1.out
aprun -n 2 -N 2 --cc depth -d 32 python pytorch_cnn_ddp.py >& pytorch_cnn_ddp.py.2.out
aprun -n 4 -N 4 --cc depth -d 16 python pytorch_cnn_ddp.py >& pytorch_cnn_ddp.py.4.out

# PyTorch data loader does not work for num_workers>0 for multiple node
aprun -n 8 -N 4 --cc depth -d 16 python pytorch_cnn_ddp.py --num_workers 0 --num_threads 0 >& pytorch_cnn_ddp.py.8.out

