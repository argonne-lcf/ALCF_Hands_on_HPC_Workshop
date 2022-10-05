#!/bin/bash -l
#PBS -l nodes=1:ppn=16
#PBS -l walltime=0:20:00
#PBS -A SDL_Workshop
#PBS -q SDL_Workshop

module load conda
conda activate

cd $PBS_O_WORKDIR
for n in 1 2 4
do  
    aprun -n $n python pytorch_cifar10.py --device gpu --epochs 32 --wandb --project cifar10_ddp >& cifar10.$n.dat
done


