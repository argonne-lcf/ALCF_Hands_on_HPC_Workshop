#!/bin/bash

export NRANKS=$(wc -l < "${PBS_NODEFILE}");
export NGPU_PER_RANK=$(nvidia-smi -L | wc -l);
export NGPUS="$((${NRANKS}*${NGPU_PER_RANK}))";
export NDEPTH=64;

echo "${NDEPTH}"

RANK_IDX=$((${PMI_RANK} % ${NRANKS}))

gpu=$((${PMI_LOCAL_RANK} % ${NGPU_PER_RANK}))
export CUDA_VISIBLE_DEVICES=$gpu
echo "RANK=${RANK_IDX}/${NRANKS} LOCAL_RANK=${PMI_LOCAL_RANK}/${NGPU_PER_RANK}, gpu=${gpu}/${NGPUS}"
exec "$@"
