#!/bin/bash

# Submit a batch job to run this job on Polaris
# Note: the command-line parameters are similar to scheduler command lines
# Note: this job will run only jobs with a matching tag
balsam queue submit \
    -n 1 -t 5 -q HandsOnHPC -A alcf_training \
    --site polaris_tutorial \
    --tag workflow=hello \
    --job-mode mpi
  
# List the Balsam BatchJob
# Note: Balsam will submit this job to PBS, so it will appear in qstat output after a short delay
balsam queue ls 

# List status of the Hello job
balsam job ls --tag workflow=hello
