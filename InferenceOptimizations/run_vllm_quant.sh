#!/bin/bash
module use /soft/modulefiles
module load conda/2024-10-30-workshop
conda activate

python3 run_vllm.py --model="neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a8" \
                    --tensor-parallel-size=4 \
                    --output-len=64 \
                    --block-size=16
