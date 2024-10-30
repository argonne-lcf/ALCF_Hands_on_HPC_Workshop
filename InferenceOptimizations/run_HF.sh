#!/bin/bash
module use /soft/modulefiles/
module load conda

conda activate vLLM_workshop

python3 run_HF.py --model="meta-llama/Meta-Llama-3-8B"
