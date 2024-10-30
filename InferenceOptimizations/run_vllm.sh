#!/bin/bash
module use /soft/modulefiles/
module load conda

conda activate vLLM_workshop

python3 run_vllm.py --model="meta-llama/Meta-Llama-3-8B" \
                    --tensor-parallel-size=4 \
                    --output-len=64 \
                    --block-size=16 \
                    --dtype="float16"
                    
