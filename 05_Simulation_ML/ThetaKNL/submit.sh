#!/bin/bash
#COBALT -n 1
#COBALT -t 00:60:00
#COBALT -q debug-cache-quad 
#COBALT -A SDL_Workshop

echo "Running Cobalt Job $COBALT_JOBID."

source setup.sh
aprun -n 1 -N 1 -e OMP_NUM_THREADS=32 -d 32 -j 2 -e KMP_BLOCKTIME=0 -cc depth ./app
