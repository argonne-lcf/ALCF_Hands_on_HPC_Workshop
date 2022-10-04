#!/bin/bash

# Submit BatchJobs at multiple sites
# Polaris
balsam queue submit \
  -n 1 -t 10 -q debug -A datascience \
  --site polaris_tutorial \
  --tag workflow=hello_multisite \
  --job-mode mpi

# thetagpu
# note: should use full-node queue
balsam queue submit \
  -n 1 -t 10 -q single-gpu -A datascience \
  --site thetagpu_tutorial \
  --tag workflow=hello_multisite \
  --job-mode mpi

# # theta knl
# balsam queue submit \
#   -n 1 -t 10 -q debug-flat-quad -A Comp_Perf_Workshop \
#   --site thetaknl_tutorial \
#   --tag workflow=hello_multisite \
#   --job-mode mpi

# # cooley
# balsam queue submit \
#   -n 1 -t 10 -q debug -A Comp_Perf_Workshop \
#   --site cooley_tutorial \
#   --tag workflow=hello_multisite \
#   --job-mode mpi

# christine laptop
balsam queue submit \
  -n 1 -t 10 -q local -A local \
  --site /Users/csimpson/my-site \
  --tag workflow=hello_multisite \
  --job-mode mpi \

# List queues
balsam queue ls
