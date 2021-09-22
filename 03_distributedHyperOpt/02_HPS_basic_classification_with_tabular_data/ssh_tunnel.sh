#!/bin/bash

# User configuration
export ALCF_USERNAME=regele
export JUPYTER_PORT=8888
export THETAGPU_NODE=thetagpu05

# port on your local machine
# picking a number other than the default 8888 is recommended
export PORT_NUM=8889
ssh -L $PORT_NUM:localhost:8888 $ALCF_USERNAME@theta.alcf.anl.gov \
    "ssh -L 8888:localhost:$JUPYTER_PORT" $THETAGPU_NODE