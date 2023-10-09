#!/bin/bash -l
#PBS -S /bin/bash
#PBS -N nekRS_train
#PBS -l walltime=00:30:00
#PBS -l select=2:ncpus=64:ngpus=4
#PBS -l filesystems=eagle:home
#PBS -k doe
#PBS -j oe
##PBS -A fallwkshp23
##PBS -q fallws23scaling
#PBS -A datascience
#PBS -q debug-scaling
#PBS -V

module load conda/2022-09-08
conda activate /eagle/projects/fallwkshp23/SmartSim/ssim
module load cudatoolkit-standalone
module load cmake
export CRAY_ACCEL_TARGET=nvidia80

export NEKRS_HOME=/eagle/projects/fallwkshp23/NekRS-ML/exe/Polaris/smartredis
export PATH=$NEKRS_HOME/bin:$PATH
export LD_LIBRARY_PATH=/eagle/projects/fallwkshp23/SmartSim/SmartRedis/install/lib:$LD_LIBRARY_PATH

python ssim_driver_polaris.py sim.executable=$NEKRS_HOME/bin/nekrs \
  run_args.simprocs=2 run_args.simprocs_pn=2 \
  train.executable=./trainer.py run_args.mlprocs=2 run_args.mlprocs_pn=2 \
  train.device=cuda train.affinity=./affinity_ml.sh
