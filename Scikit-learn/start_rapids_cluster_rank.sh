#start_rapids_cluster_rank.sh

ROLE=$1
HOSTNAME=$HOSTNAME

# NVLINK, IB, or TCP (default TCP)
CLUSTER_MODE="TCP"

MAX_SYSTEM_MEMORY=$(free -m | awk '/^Mem:/{print $2}')M
DEVICE_MEMORY_LIMIT="29GB"
POOL_SIZE="31GB"
# A100 big mem
# DEVICE_MEMORY_LIMIT="70GB"
# POOL_SIZE="78GB"

# Used for writing scheduler file to shared storage
LOCAL_DIRECTORY=$HOME/dask-local-directory
SCHEDULER_FILE=$LOCAL_DIRECTORY/scheduler.json
LOGDIR="$LOCAL_DIRECTORY/logs"
WORKER_DIR="/tmp/dask-workers/"
DASHBOARD_PORT=8787

# Purge Dask worker and log directories
if [ "$ROLE" = "SCHEDULER" ]; then
    rm -rf $LOGDIR/*
    mkdir -p $LOGDIR
    rm -rf $WORKER_DIR/*
    mkdir -p $WORKER_DIR
fi

# Purge Dask config directories
rm -rf ~/.config/dask


# Dask/distributed configuration
export DASK_DISTRIBUTED__COMM__TIMEOUTS__CONNECT="100s"
export DASK_DISTRIBUTED__COMM__TIMEOUTS__TCP="600s"
export DASK_DISTRIBUTED__COMM__RETRY__DELAY__MIN="1s"
export DASK_DISTRIBUTED__COMM__RETRY__DELAY__MAX="60s"
export DASK_DISTRIBUTED__WORKER__MEMORY__Terminate="False"


# Setup scheduler
if [ "$ROLE" = "SCHEDULER" ]; then

  if [ "$CLUSTER_MODE" = "NVLINK" ]; then
     CUDA_VISIBLE_DEVICES='0' DASK_UCX__CUDA_COPY=True DASK_UCX__TCP=True DASK_UCX__NVLINK=True DASK_UCX__INFINIBAND=False DASK_UCX__RDMACM=False nohup dask-scheduler --dashboard-address $DASHBOARD_PORT --protocol ucx --scheduler-file $SCHEDULER_FILE > $LOGDIR/$HOSTNAME-scheduler.log 2>&1 &
  fi

  if [ "$CLUSTER_MODE" = "IB" ]; then
     DASK_RMM__POOL_SIZE=1GB CUDA_VISIBLE_DEVICES='0' DASK_UCX__CUDA_COPY=True DASK_UCX__TCP=True DASK_UCX__NVLINK=True DASK_UCX__INFINIBAND=True DASK_UCX__RDMACM=False UCX_NET_DEVICES=mlx5_0:1 nohup dask-scheduler --dashboard-address $DASHBOARD_PORT --protocol ucx --interface ib0 --scheduler-file $SCHEDULER_FILE > $LOGDIR/$HOSTNAME-scheduler.log 2>&1 &
  fi

  if [ "$CLUSTER_MODE" = "TCP" ]; then
     CUDA_VISIBLE_DEVICES='0' nohup dask-scheduler --dashboard-address $DASHBOARD_PORT --protocol tcp --scheduler-file $SCHEDULER_FILE > $LOGDIR/$HOSTNAME-scheduler.log 2>&1 &
  fi
fi


# Setup workers
if [ "$CLUSTER_MODE" = "NVLINK" ]; then
    dask-cuda-worker --device-memory-limit $DEVICE_MEMORY_LIMIT --local-directory $LOCAL_DIRECTORY --rmm-pool-size=$POOL_SIZE --memory-limit=$MAX_SYSTEM_MEMORY --enable-tcp-over-ucx --enable-nvlink  --disable-infiniband --scheduler-file $SCHEDULER_FILE >> $LOGDIR/$HOSTNAME-worker.log 2>&1
fi

if [ "$CLUSTER_MODE" = "IB" ]; then
    dask-cuda-worker --device-memory-limit $DEVICE_MEMORY_LIMIT --local-directory $LOCAL_DIRECTORY --rmm-pool-size=$POOL_SIZE --memory-limit=$MAX_SYSTEM_MEMORY --enable-tcp-over-ucx --enable-nvlink  --enable-infiniband --disable-rdmacm --scheduler-file $SCHEDULER_FILE >> $LOGDIR/$HOSTNAME-worker.log 2>&1
fi

if [ "$CLUSTER_MODE" = "TCP" ]; then
    dask-cuda-worker --device-memory-limit $DEVICE_MEMORY_LIMIT --local-directory $LOCAL_DIRECTORY --rmm-pool-size=$POOL_SIZE --memory-limit=$MAX_SYSTEM_MEMORY --scheduler-file $SCHEDULER_FILE >> $LOGDIR/$HOSTNAME-worker.log 2>&1
fi
