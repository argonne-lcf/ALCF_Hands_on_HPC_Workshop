#!/bin/bash

module swap PrgEnv-intel PrgEnv-gnu
module load cmake
export CRAYPE_LINK_TYPE=dynamic
