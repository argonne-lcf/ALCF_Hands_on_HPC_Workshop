#!/bin/bash
module use /soft/modulefiles
module load conda/2024-10-30-workshop
conda activate

python3 run_HF.py --model="meta-llama/Meta-Llama-3-8B"
