#!/bin/bash

# Set run env
nodes=`wc -l < $PBS_NODEFILE`
striping_unit=16777216
max_striping_factor=128
let striping_factor=$nodes/2
if [ $striping_factor -gt $max_striping_factor ]; then
  striping_factor=$max_striping_factor
fi
if [ $striping_factor -lt 1 ]; then
  striping_factor=1
fi
MPICH_MPIIO_HINTS="*:striping_unit=${striping_unit}:striping_factor=${striping_factor}:romio_cb_write=enable:romio_ds_write=disable:romio_no_indep_rw=true"

ulimit -s unlimited
export NEKRS_GPU_MPI=1
export MPICH_MPIIO_HINTS=$MPICH_MPIIO_HINTS
export MPICH_MPIIO_STATS=1
export NEKRS_CACHE_BCAST=1
export NEKRS_LOCAL_TMP_DIR=/local/scratch
export MPICH_GPU_SUPPORT_ENABLED=1
#export MPICH_OFI_NIC_POLICY=NUMA
export FI_OFI_RXM_RX_SIZE=32768

# Run the driver script
sim_arguments="--setup turbChannel.par --backend CUDA --device-id 0"
python ssim_driver_polaris.py \
  sim.executable=$NEKRS_HOME/bin/nekrs run_args.simprocs=2 run_args.simprocs_pn=2 \
  sim.arguments="${sim_arguments}" sim.affinity=./affinity_nrs.sh \
  train.executable=./trainer.py run_args.mlprocs=2 run_args.mlprocs_pn=2 \
  train.device=cuda train.affinity=./affinity_ml.sh
