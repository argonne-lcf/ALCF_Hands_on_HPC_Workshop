#!/bin/bash
module use /soft/modulefiles/
module load conda

conda create -n vLLM_workshop python=3.11 -y
