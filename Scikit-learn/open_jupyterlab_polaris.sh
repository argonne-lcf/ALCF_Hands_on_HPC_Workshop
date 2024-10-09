#!/bin/bash
# open_jupyterlab_polaris.sh

NODE_ADDRESS=${1}
RAPIDS_WORKDIR='~'
SSH_MULTIPLEX=polaris
PORT=8881
PORTD=8787
#ssh polaris "ps -ef | grep jupyter | grep -v grep | awk -F ' ' '{print \$2}' | xargs kill -9 2>/dev/null; rm ~/jupyter_pol.log" 2>/dev/null
ssh polaris "ssh ${NODE_ADDRESS} \"echo \\\$(hostname) | tee ~/jupyter_pol.log && \
source ${RAPIDS_WORKDIR}/activate_rapids_env_polaris.sh 2> /dev/null && \
nohup jupyter lab --no-browser --port=${PORT} &>> ~/jupyter_pol.log & \
JPYURL=''; while [ -z \\\${JPYURL} ]; do sleep 2; JPYURL=\\\$(sed -n '/] http:\/\/localhost/p' ~/jupyter_pol.log | sed 's/^.*\(http.*\)$/\1/g'); done; echo \\\${JPYURL}\" " > ~/jupyter_pol.log & \
PORT=''; while [ -z ${PORT} ]; do sleep 2; PORT=$(sed -n 's/.*:\([0-9][0-9]*\)\/.*/\1/p' ~/jupyter_pol.log); done && \
#ssh -O forward -L ${PORT}:localhost:${PORT} polaris && \
echo "Open this url $(grep token ~/jupyter_pol.log)"
ssh -O forward -L ${PORT}:localhost:${PORT} -L ${PORTD}:localhost:${PORTD} polaris && \
ssh -t polaris ssh -t -L ${PORT}:localhost:${PORT} -L ${PORTD}:localhost:${PORTD} ${NODE_ADDRESS}
