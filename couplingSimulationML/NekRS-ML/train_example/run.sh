#!/bin/bash

python ssim_driver_polaris.py sim.executable=$NEKRS_HOME/bin/nekrs run_args.simprocs=1 run_args.simprocs_pn=1 train.executable=./trainer.py run_args.mlprocs=1 run_args.mlprocs_pn=1 train.device=cuda train.affinity=./affinity_ml.sh
