#!/bin/bash

module use /soft/modulefiles
module load daos/base
module load frameworks

DAOS_POOL=datascience
DAOS_CONT=test_container
launch-dfuse.sh ${DAOS_POOL}:${DAOS_CONT}

NNODES=`wc -l < $PBS_NODEFILE`
RANKS_PER_NODE=12          # Number of MPI ranks per node
NRANKS=$(( NNODES * RANKS_PER_NODE ))
echo "NUM_OF_NODES=${NNODES}  TOTAL_NUM_RANKS=${NRANKS}  RANKS_PER_NODE=${RANKS_PER_NODE}"
CPU_BINDING=list:4:9:14:19:20:25:56:61:66:71:74:79

mpiexec -np ${NRANKS} --ppn ${RANKS_PER_NODE} \
    --cpu-bind=${CPU_BINDING}  \
    --no-vni -genvall \
    python -c "import torch; torch.save(torch.randn((10,)), f'/tmp/${DAOS_POOL}/${DAOS_CONT}/randn_{torch.randint(1000, (1,)).item()}.pt')"

ls -l /tmp/${DAOS_POOL}/${DAOS_CONT}
clean-dfuse.sh ${DAOS_POOL}:${DAOS_CONT}
