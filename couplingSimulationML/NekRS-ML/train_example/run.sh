#!/bin/bash

python ssim_driver_polaris.py sim.executable=$NEKRS_HOME/bin/nekrs run_args.simprocs=2 run_args.simprocs_pn=2 train.executable=./trainer.py run_args.mlprocs=2 run_args.mlprocs_pn=2 train.device=cuda train.affinity=./affinity_ml.sh
