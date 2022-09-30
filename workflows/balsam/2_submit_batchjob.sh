#!/bin/bash

# Submit a batch job to run this job on ThetaGPU
# Note: the command-line parameters are similar to scheduler command lines
# Note: this job will run only jobs with a matching tag
balsam queue submit \
    -n 1 -t 10 -q training-gpu -A Comp_Perf_Workshop \
    --site thetagpu_tutorial \
    --tag workflow=hello \
    --job-mode mpi
  
# List the Balsam BatchJob
# Note: Balsam will submit this job to Cobalt, so it will appear in qstat output after a short delay
balsam queue ls 

# List status of the Hello job
balsam job ls --tag workflow=hello
