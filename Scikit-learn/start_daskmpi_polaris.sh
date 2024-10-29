#!/bin/bash

NRANKS_PER_NODE=32
NUM_NODES=$(cat $PBS_NODEFILE | wc -l)
NTOTRANKS=$(( NUM_NODES * NRANKS_PER_NODE ))

TMP_EXE=tmp_rpds.sh

cat > ${TMP_EXE} << EOF
#!/bin/bash
if [ \$PMI_RANK == 0 ]; then
    ~/activate_rapids_env_polaris.sh ~/start_daskmpi_rank.sh SCHEDULER
else
    ~/activate_rapids_env_polaris.sh ~/start_daskmpi_rank.sh
fi
EOF

chmod 755 ${TMP_EXE}
sleep 5

mpiexec -n $NTOTRANKS --ppn $NRANKS_PER_NODE ./${TMP_EXE}

rm ./${TMP_EXE}
