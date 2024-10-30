#!/bin/bash
module use /soft/modulefiles/
module load conda

conda activate vLLM_workshop

python3 run_vllm.py --model="meta-llama/Meta-Llama-3-8B" \
                    --speculative-model="turboderp/Qwama-0.5B-Instruct" \
                    --tensor-parallel-size=4 \
                    --speculative-draft-tensor-parallel-size=1 \
                    --num-speculative-tokens=5 \
                    --output-len=64 \
                    --block-size=16 \
                    --dtype="float16"
