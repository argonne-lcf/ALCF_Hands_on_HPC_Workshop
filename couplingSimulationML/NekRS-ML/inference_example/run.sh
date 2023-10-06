#!/bin/bash

python ssim_driver_polaris.py sim.executable=$NEKRS_HOME/bin/nekrs run_args.simprocs=3 run_args.simprocs_pn=3 inference.model_path=./model_jit.pt inference.device=GPU:3
