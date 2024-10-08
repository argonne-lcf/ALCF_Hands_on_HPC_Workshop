#!/bin/bash
module use /soft/modulefiles && module load conda/2024-04-29 && conda activate /lus/grand/projects/alcf_training/rapids/polaris/rapids-23.04_polaris && \
export LD_LIBRARY_PATH=/lus/grand/projects/alcf_training/rapids/polaris/rapids-23.04_polaris/targets/x86_64-linux/lib:${LD_LIBRARY_PATH} && \
