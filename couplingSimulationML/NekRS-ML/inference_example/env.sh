#!/bin/bash

module load conda/2022-09-08
conda activate /eagle/projects/fallwkshp23/SmartSim/ssim
module load cudatoolkit-standalone
module load cmake
export CRAY_ACCEL_TARGET=nvidia80

export NEKRS_HOME=/eagle/projects/fallwkshp23/NekRS-ML/exe/Polaris/smartredis
export PATH=$NEKRS_HOME/bin:$PATH
export LD_LIBRARY_PATH=/eagle/projects/fallwkshp23/SmartSim/SmartRedis/install/lib:$LD_LIBRARY_PATH


