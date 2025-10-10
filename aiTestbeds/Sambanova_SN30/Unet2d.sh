#! /bin/bash 
set -e
if [ $# -ne 5 ] ; then
  echo $#, $1, $2, $3, $4, $5
  echo "Unet2d.sh {compile,pcompile,run,prun} Image_size Batch_size Number_of_instances RunID"
  exit 1
fi
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
if [ ! -e ${OUTDIR} ] ; then
mkdir -p ${OUTDIR}
fi
########################################
SECONDS=0
BS=$3
NP=$4
NUM_WORKERS=4
NUM_TILES=4
DS=/data/ANL/kaggle_3m
CACHE_DIR=/data/scratch/${USER}/kaggle_3m_${2}
if [ ! -d ${CACHE_DIR} ] ; then
  mkdir -p ${CACHE_DIR}
fi
export OMP_NUM_THREADS=16
# The base python env now supports Unet2D, including batch mode
#if [ -e /opt/sambaflow/apps/image/segmentation/venv/bin/activate ] ; then
#  source /opt/sambaflow/apps/image/segmentation/venv/bin/activate
#  else
#  source /opt/sambaflow/venv/bin/activate
#fi
if [ -e /opt/sambaflow/apps/image/unet ] ; then
    UNET=/opt/sambaflow/apps/image/unet
elif [ -e /opt/sambaflow/apps/vision/segmentation/compile.py ] ; then
    UNET=/opt/sambaflow/apps/vision/segmentation/
elif [ -e /opt/sambaflow/apps/image/segmentation ] ; then
    UNET=/opt/sambaflow/apps/image/segmentation/
else
    echo "Cannot find UNET"
    exit
fi
HD=${2}
if [ ${HD} == "1024" ] ; then
  HD=1k
elif [ ${HD} == "2048" ] ; then
  HD=2k
elif [ ${HD} == "4096" ] ; then
  HD=4k
fi

echo "Model: UNET2d" >> ${OUTPUT_PATH} 2>&1
echo "Date: " $(date +%m/%d/%y) >> ${OUTPUT_PATH} 2>&1
echo "Time: " $(date +%H:%M) >> ${OUTPUT_PATH} 2>&1

#rm -rf log_dir*

if [ "${1}" == "compile" ] ; then
   #compile loop
   echo "COMPILE" >> ${OUTPUT_PATH} 2>&1
     if [ -e ${OUTDIR}/unet_train_${BS}_${2}_single/unet_train_${BS}_${2}_single_${NUM_TILES}.pef ] ; then
       rm ${OUTDIR}/unet_train_${BS}_${2}_single/unet_train_${BS}_${2}_single_${NUM_TILES}.pef
     fi
     if [ -e ${UNET}/compile.py ] ; then
       COMMAND="python ${UNET}/compile.py compile --init-features 32 --in-channels=3 --num-classes 2 --num-flexible-classes 1 --in-width=${2} --in-height=${2} --batch-size=${BS} --enable-conv-tiling --mac-human-decision /opt/sambaflow/apps/vision/segmentation/jsons/hd_files/hd_unet_256_depth2colb.json --compiler-configs-file /opt/sambaflow/apps/vision/segmentation/jsons/compiler_configs/unet_compiler_configs_depth2colb.json --enable-stoc-rounding --num-tiles=4 --pef-name=unet_train_${BS}_${2}_single_${NUM_TILES} --output-folder=${OUTDIR}"	 
     else
#old
       COMMAND="python ${UNET}/unet.py compile -b ${BS} --in-channels=${NUM_WORKERS} --in-width=${2} --in-height=${2} --enable-conv-tiling --mac-v2 --mac-human-decision ${UNET}/jsons/hd_files/hd_unet_${HD}_tgm.json --compiler-configs-file ${UNET}/jsons/compiler_configs/unet_compiler_configs_no_inst.json --pef-name="unet_train_${BS}_${2}_single" > compile_${BS}_${2}_single.log 2>&1"
     fi
     echo $COMMAND >> ${OUTPUT_PATH} 2>&1
     eval $COMMAND >> ${OUTPUT_PATH} 2>&1

elif [ "${1}" == "pcompile" ] ; then
  #parallel
   echo "Parallel compile" >> ${OUTPUT_PATH} 2>&1
   #BS=$((BS/NP))
   if [ -e ${OUTDIR}/unet_train_${BS}_${2}_NP_${NUM_TILES}/unet_train_${BS}_${2}_NP_${NUM_TILES}.pef ] ; then
     rm ${OUTDIR}/unet_train_${BS}_${2}_NP_${NUM_TILES}/unet_train_${BS}_${2}_NP_${NUM_TILES}.pef
   fi
   if [ -e ${UNET}/hook.py ] ; then
     COMMAND="python /opt/sambaflow/apps/vision/segmentation/compile.py compile --init-features 32 --in-channels=3 --num-classes 2 --num-flexible-classes 1 --in-width=${2} --in-height=${2} --batch-size=${BS} --enable-conv-tiling --mac-human-decision /opt/sambaflow/apps/vision/segmentation/jsons/hd_files/hd_unet_256_depth2colb.json --compiler-configs-file /opt/sambaflow/apps/vision/segmentation/jsons/compiler_configs/unet_compiler_configs_depth2colb.json --num-tiles=4 --pef-name=unet_train_${BS}_${2}_NP_${NUM_TILES}  --data-parallel -ws 2 --output-folder=${OUTDIR}"
   else
     COMMAND="python ${UNET}/unet.py compile -b ${BS} --in-channels=${NUM_WORKERS} --in-width=${2} --in-height=${2} --enable-conv-tiling --mac-v2 --mac-human-decision ${UNET}/jsons/hd_files/hd_unet_${HD}_tgm.json --compiler-configs-file ${UNET}/jsons/compiler_configs/unet_compiler_configs_no_inst.json --pef-name=unet_train_${BS}_${2}_NP  --data-parallel -ws 2 --output-folder=${OUTDIR}"
   fi
   echo $COMMAND >> ${OUTPUT_PATH} 2>&1
   eval $COMMAND >> ${OUTPUT_PATH} 2>&1

elif [ "${1}" == "run" ] ; then
 #single
   echo "RUN" >> ${OUTPUT_PATH} 2>&1
   export OMP_NUM_THREADS=16
   export SF_RNT_NUMA_BIND=2
   export SF_RNT_FSM_POLL_BUSY_WAIT=1
   export SF_RNT_DMA_POLL_BUSY_WAIT=1
   #run single 
   if [ -e ${UNET}/hook.py ] ; then
    COMMAND="srun --nodelist $(hostname) python /opt/sambaflow/apps/vision/segmentation//hook.py run --data-cache=${CACHE_DIR}  --data-in-memory --num-workers=${NUM_WORKERS} --enable-tiling  --min-throughput 395 --in-channels=3 --in-width=${2} --in-height=${2} --init-features 32 --batch-size=${BS} --max-epochs 10 --data-dir ${DS} --log-dir log_dir_unet_${2}_${BS}_single_${NUM_TILES} --pef=${OUTDIR}/unet_train_${BS}_${2}_single_${NUM_TILES}/unet_train_${BS}_${2}_single_${NUM_TILES}.pef"

   else
     COMMAND="srun --nodelist $(hostname) python ${UNET}/unet_hook.py  run --num-workers=${NUM_WORKERS} --do-train --in-channels=3 --in-width=${2} --in-height=${2} --init-features 32 --batch-size=${BS} --epochs 10  --data-dir ${DS} --log-dir log_dir_unet_${2}_${3} --pef=${OUTDIR}/unet_train_${BS}_${2}_single/unet_train_${BS}_${2}_single.pef --use-sambaloader"
   fi
   echo $COMMAND >> ${OUTPUT_PATH} 2>&1
   eval $COMMAND >> ${OUTPUT_PATH} 2>&1
    #end run single

elif [ "${1}" == "prun" ] ; then
  #Parallel
  #BS=$((BS/NP))
  echo "PRUN" >> ${OUTPUT_PATH} 2>&1
  echo "NP=${NP}" >> ${OUTPUT_PATH} 2>&1
  cpus=$((128/NP))
  COMMAND="sbatch --gres=rdu:1 --tasks-per-node ${NP} --nodes 1 --nodelist $(hostname) --cpus-per-task=${cpus} $(pwd)/unet_batch.sh ${NP} ${NUM_WORKERS} ${BS} ${2} ${5}"
   echo $COMMAND >> ${OUTPUT_PATH} 2>&1
   eval $COMMAND >> ${OUTPUT_PATH} 2>&1
fi
echo "Duration: " $SECONDS >> ${OUTPUT_PATH} 2>&1
