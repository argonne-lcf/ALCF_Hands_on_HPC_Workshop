#! /bin/bash
set -e
LOGDIR=`date +%m%d%y.%H`
if [ "$5" ] ; then
LOGDIR=$5
fi
MODEL_NAME="Unet2d"
OUTPUT_PATH=/data/ANL/results/$(hostname)/${USER}/${LOGDIR}/${MODEL_NAME}.out
echo "Using ${OUTPUT_PATH} for output"
mkdir -p /data/ANL/results/$(hostname)/${USER}/${LOGDIR}

#######################
export OMP_NUM_THREADS=8
#######################
# Start script timer
SECONDS=0
# Temp file location
OUTDIR=$(pwd)/out/${MODEL_NAME}

BS=$3
IM=$4
DS=/data/ANL/kaggle_3m
CACHE_DIR=/data/scratch/${USER}/kaggle_3m_${IM}
if [ ! -d ${CACHE_DIR} ] ; then
  mkdir -p ${CACHE_DIR}
fi
NUM_WORKERS=${2}
NP=${1}
export OMP_NUM_THREADS=16

# No need to source these; base env is sufficient. 
#if [ -e /opt/sambaflow/apps/image/segmentation/venv/bin/activate ] ; then
#source /opt/sambaflow/apps/image/segmentation/venv/bin/activate
#else
#source /opt/sambaflow/venv/bin/activate
#fi

if [ -e /opt/sambaflow/apps/image/unet ] ; then
  UNET=/opt/sambaflow/apps/image/unet
elif [ -e /opt/sambaflow/apps/image/segmentation ] ; then
  UNET=/opt/sambaflow/apps/vision/segmentation/
else
  echo "Cannot find UNET"
  exit
fi

echo "Model: ${MODEL_NAME} " > ${OUTPUT_PATH} 2>&1
echo "Date: " $(date +%m/%d/%y) >> ${OUTPUT_PATH} 2>&1
echo "Time: " $(date +%H:%M) >> ${OUTPUT_PATH} 2>&1

#    export SAMBA_CCL_USE_PCIE_TRANSPORT=0
    export SF_RNT_NUMA_BIND=2
    export SF_RNT_FSM_POLL_BUSY_WAIT=1
    export SF_RNT_DMA_POLL_BUSY_WAIT=1
#    export SF_RNT_LOG_LEVEL=DEBUG
    #export CCL_TIMEOUT=30
    #export SF_RNT_TILE_AFFINITY=0xf0000000
    export SAMBA_CCL_USE_PCIE_TRANSPORT=1
    export SF_RNT_NUMA_BIND=2
    export SF_RNT_FSM_POLL_BUSY_WAIT=1
    export SF_RNT_DMA_POLL_BUSY_WAIT=1
    rm -rf log_dir_unet_${NP}_train_kaggle
    if [ -e ${UNET}/hook.py ] ; then
        COMMAND="srun --mpi=pmi2 python /opt/sambaflow/apps/vision/segmentation/hook.py run --data-cache=${CACHE_DIR}  --data-in-memory --num-workers=${NUM_WORKERS} --enable-tiling  --min-throughput 395 --in-channels=3 --in-width=${IM} --in-height=${IM} --init-features 32 --batch-size=${BS} --max-epochs 10 --data-dir ${DS} --log-dir log_dir_unet_${IM}_${BS}_${NP} --data-parallel --reduce-on-rdu --pef=${OUTDIR}/unet_train_${BS}_${IM}_NP_4/unet_train_${BS}_${IM}_NP_4.pef"
    else
        COMMAND="srun --mpi=pmi2 python ${UNET}/unet_hook.py  run --data-cache-dir ${CACHE_DIR} --num-workers=${NUM_WORKERS} --do-train --in-channels=3 --in-width=${IM} --in-height=${IM} --init-features 32 --batch-size=${BS} --epochs 10  --data-dir ${DS} --log-dir log_dir_unet_${IM}_${BS}_${NP} --pef=$(pwd)/out/unet_train_${BS}_${IM}_NP/unet_train_${BS}_${IM}_NP.pef --data-parallel --reduce-on-rdu --use-sambaloader > run_unet_${BS}_${IM}_${NP}.log 2>&1"
    fi
echo $COMMAND >> ${OUTPUT_PATH} 2>&1
eval $COMMAND >> ${OUTPUT_PATH} 2>&1
echo "Duration: " $SECONDS >> ${OUTPUT_PATH} 2>&1                                                              
