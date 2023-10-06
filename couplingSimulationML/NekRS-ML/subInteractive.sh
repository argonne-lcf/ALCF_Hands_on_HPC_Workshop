#!/bin/bash

runtime=01:00:00
project=datascience
queue=debug-scaling
#project=fallwkshp23
#queue=fallws23single
nodes=1

qsub -I -l select=$nodes:ncpus=64:ngpus=4,walltime=$runtime,filesystems=home:eagle:grand -q $queue -A $project
