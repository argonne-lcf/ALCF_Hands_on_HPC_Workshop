#!/bin/bash
## This wrapper should be used with nsys profiler to trace in the case of larger than
## 2 Nodes. The script is set up to trace rank 0 of first 2 Nodes in the case of
## profiling a job running on larger than 2 nodes.
FNAME_EXT=$(basename "$2")
FNAME="${FNAME_EXT%%.*}"
 
NNODES=`wc -l < $PBS_NODEFILE`

WORK_DIR=/home/hossainm/hpc_workshop_october_2024
DTAG=$(date +%F_%H%M%S)
PROFILER_OUTDIR=${WORK_DIR}/profiles/nsys_seq_parallel_bf16_n${NNODES}_${DTAG}/${FNAME}_n${NNODES}_${DTAG}
RUN_ID=nsys_seq_parallel_bf16_n${NNODES}_${DTAG}

mkdir -p ${PROFILER_OUTDIR}
NSYS_OPTS=" -o ${PROFILER_OUTDIR}/${RUN_ID}_%q{PMI_RANK} --stats=true --show-output=true "
 
PROFRANK=0
RANKCUTOFF=8

if [[ $PALS_LOCAL_RANKID -eq $PROFRANK ]] && [[ $PMI_RANK -lt $RANKCUTOFF ]]; then
  echo "On rank ${PMI_RANK}, collecting traces "
  nsys profile $NSYS_OPTS "$@"
else
  "$@"
fi
