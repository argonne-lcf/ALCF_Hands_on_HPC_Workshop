#!/bin/bash
set -e

Cleanup () {
if ps -p $DBPID > /dev/null; then
	kill -15 $DBPID
fi
}

trap Cleanup exit

/eagle/projects/fallwkshp23/SmartSim/ssim/bin/python -m smartsim._core.entrypoints.colocated +lockfile smartsim-c7beff5.lock +db_cpus 4 +ifname lo +command /lus/eagle/projects/fallwkshp23/SmartSim/SmartSim/smartsim/_core/bin/redis-server /lus/eagle/projects/fallwkshp23/SmartSim/SmartSim/smartsim/_core/config/redis6.conf --loadmodule /lus/eagle/projects/fallwkshp23/SmartSim/SmartSim/smartsim/_core/lib/redisai.so THREADS_PER_QUEUE 4 INTER_OP_PARALLELISM 1 INTRA_OP_PARALLELISM 1 --port 6780 --logfile /dev/null --maxclients 100000 --cluster-node-timeout 30000 &
DBPID=$!

taskset -c 0-$(nproc --ignore=5) $@

