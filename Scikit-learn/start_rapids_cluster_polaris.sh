#!/bin/bash

# start_rapids_cluster_polaris.sh

NUM_NODES=$(cat $PBS_NODEFILE | wc -l)
TMP_EXE=tmp_rpds.sh

cat > ${TMP_EXE} << EOF
#!/bin/bash
if [ \$PMI_RANK == 0 ]; then
    ~/activate_rapids_env_polaris.sh ~/start_rapids_cluster_rank.sh SCHEDULER
else
    ~/activate_rapids_env_polaris.sh ~/start_rapids_cluster_rank.sh
fi
EOF

chmod 755 ${TMP_EXE}
sleep 5

mpiexec -n $NUM_NODES --ppn 1 ~/${TMP_EXE}

rm ~/${TMP_EXE}
