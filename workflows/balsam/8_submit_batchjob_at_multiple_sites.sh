#!/bin/bash

# Submit BatchJobs at multiple sites
# Polaris
balsam queue submit \
  -n 1 -t 10 -q debug -A datascience \
  --site polaris-testing \
  --tag workflow=vec_multisite \
  --job-mode mpi

# thetagpu
balsam queue submit \
  -n 1 -t 10 -q single-gpu -A datascience \
  --site thetagpu_tutorial \
  --tag workflow=vec_multisite \
  --job-mode mpi

# List queues
balsam queue ls
