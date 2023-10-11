#!/bin/bash -l
#PBS -l walltime=00:30:00
#PBS -l nodes=2:ppn=4
#PBS -N cnn_ds
#PBS -k doe
#PBS -j oe
#PBS -A fallwkshp23
#PBS -q fallws23scaling
#PBS -l filesystems=home:eagle:grand

cd $PBS_O_WORKDIR

module load conda/2023-10-04; conda activate

aprun -n 1 -N 1 --cc depth -d 64 python pytorch_cnn_ds.py --deepspeed_config ds_config.json >& pytorch_cnn_ds.py.1.out
aprun -n 2 -N 2 --cc depth -d 32 python pytorch_cnn_ds.py --deepspeed_config ds_config.json >& pytorch_cnn_ds.py.2.out
aprun -n 4 -N 4 --cc depth -d 16 python pytorch_cnn_ds.py --deepspeed_config ds_config.json >& pytorch_cnn_ds.py.4.out
# PyTorch Data Loader does not work for num_workers>0 on multiple node
aprun -n 8 -N 4 --cc depth -d 16 python pytorch_cnn_ds.py --deepspeed_config ds_config.json --num_workers 0 --num_threads 0 >& pytorch_cnn_ds.py.8.out
