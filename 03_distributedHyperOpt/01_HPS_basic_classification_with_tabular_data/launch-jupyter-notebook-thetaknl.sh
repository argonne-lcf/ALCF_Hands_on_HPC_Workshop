#!/bin/bash

source init-dh-environment.sh

jupyter notebook &

sleep 5

# User configuration
# picking a number other than the default 8888 is recommended
export JUPYTER_PORT=8888
# port on your local machine
export LOCAL_PORT=8888

printf "\e[0;31mExecute the following command from your laptop: \e[m \n"
printf "\e[0;32mssh -tt -L $LOCAL_PORT:localhost:8888 -L 8265:localhost:8265 $USER@theta.alcf.anl.gov \"ssh -L 8888:localhost:$JUPYTER_PORT -L 8265:localhost:8265 -J thetamom3 $HOSTNAME\"\e[m\n"
