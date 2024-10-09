#!/bin/bash

# close_jupyterlab_polaris.sh

RAPIDS_WORKDIR='~'
SSH_MULTIPLEX="-S ~/.ssh/multiplex:polaris.rapids YourUsername@polaris.alcf.anl.gov"  && \
PORT=$(sed -n 's/.*:\([0-9][0-9]*\)\/.*/\1/p' ~/jupyter_pol.log)  && \
RUNNING_ON=$(head -1 ~/jupyter_pol.log)  && \
ssh -O cancel -L $PORT:localhost:$PORT ${SSH_MULTIPLEX}  && \
ssh ${SSH_MULTIPLEX} "ssh ${RUNNING_ON} \"ps -ef | grep jupyter | grep -v grep | awk -F ' ' '{print \\\$2}' | xargs kill -9  2>/dev/null &&  rm ~/jupyter_pol.log\"" && \
rm ~/jupyter_pol.log
