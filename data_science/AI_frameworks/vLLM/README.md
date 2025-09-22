# Inference with vLLM on Aurora 

vLLM is an open-source library designed to optimize the inference and serving. Originally developed at UC Berkeley's Sky Computing Lab, it has evolved into a community-driven project. The library is built around the innovative PagedAttention algorithm, which significantly improves memory management by reducing waste in Key-Value (KV) cache memory.

## Install vLLM 

The instructions below should be **run directly from a compute node**. 


```bash linenums="1" title="Install vLLM using pre-built wheels"

module load frameworks
conda create --name vllm python=3.10 -y
conda activate vllm

module unload oneapi/eng-compiler/2024.07.30.002
module use /opt/aurora/24.180.3/spack/unified/0.8.0/install/modulefiles/oneapi/2024.07.30.002
module use /soft/preview/pe/24.347.0-RC2/modulefiles
module add oneapi/release

pip install /flare/datasets/softwares/vllm-install/wheels/*
pip install /flare/datasets/softwares/vllm-install/vllm-0.6.6.post2.dev28+g5dbf8545.d20250129.xpu-py3-none-any.whl
```

## Access Model Weights

Model weights for commonly used open-weight models are downloaded and available in the following directory on Aurora:
```bash linenums="1"
/flare/datasets/model-weights/hub
```

To ensure your workflows utilize the preloaded model weights and datasets, update the following environment variables in your session. Some models hosted on Hugging Face may be gated, requiring additional authentication. To access these gated models, you will need a [Hugging Face authentication token](https://huggingface.co/docs/hub/en/security-tokens).

```bash linenums="1"
export HF_HOME="/flare/datasets/model-weights/hub"
export HF_DATASETS_CACHE="/flare/datasets/model-weights/hub"
export HF_TOKEN="YOUR_HF_TOKEN"
export RAY_TMPDIR="/tmp"
export TMPDIR="/tmp"
```

## Common Configuration Recommendations 

For small models that fit within a single tile's memory (64 GB), no additional configuration is required to serve the model. Simply set `TP=1` (Tensor Parallelism). This configuration ensures the model is run on a single tile without the need for distributed setup. Models with fewer than 7 billion parameters typically fit within a single tile. 

To utilize multiple tiles for larger models (`TP>1`), additional setup is required. This involves configuring a Ray cluster and setting the `ZE_FLAT_DEVICE_HIERARCHY` environment variable:
```bash linenums="1"
export ZE_FLAT_DEVICE_HIERARCHY=FLAT

export VLLM_HOST_IP=$(getent hosts $(hostname).hsn.cm.aurora.alcf.anl.gov | awk '{ print $1 }' | tr ' ' '\n' | sort | head -n 1)
export tiles=12
ray --logging-level debug start --head --verbose --node-ip-address=$VLLM_HOST_IP --port=6379 --num-cpus=64 --num-gpus=$tiles&
```

## Serve Small Models 

#### Using Single Tile

The following command serves `meta-llama/Llama-2-7b-chat-hf` on a single tile of a single node:
```bash linenums="1"
vllm serve meta-llama/Llama-2-7b-chat-hf --port 8000 --device xpu --dtype float16
```

#### Using Multiple Tiles

The following commands set up the Ray cluster and serve `meta-llama/Llama-3.3-70b-chat-hf` on 8 tiles on single node. 
```bash linenums="1"
export VLLM_HOST_IP=$(getent hosts $(hostname).hsn.cm.aurora.alcf.anl.gov | awk '{ print $1 }' | tr ' ' '\n' | sort | head -n 1)
ray --logging-level debug start --head --verbose --node-ip-address=$VLLM_HOST_IP --port=6379 --num-cpus=64 --num-gpus=8&
vllm serve meta-llama/Llama-2-7b-chat-hf --port 8000 --tensor-parallel-size 8 --device xpu --dtype float16 --trust-remote-code
```

## Serve Medium Models 

#### Using Single Node

These commands set up a Ray cluster and serves `meta-llama/Llama-3.3-70B-Instruct` on 8 tiles on single node. Models with up to 70 billion parameters can usually fit within a single node, utilizing multiple tiles.

```bash linenums="1"
export VLLM_HOST_IP=$(getent hosts $(hostname).hsn.cm.aurora.alcf.anl.gov | awk '{ print $1 }' | tr ' ' '\n' | sort | head -n 1)
ray --logging-level debug start --head --verbose --node-ip-address=$VLLM_HOST_IP --port=6379 --num-cpus=64 --num-gpus=8&
vllm serve meta-llama/Llama-3.3-70B-Instruct --port 8000 --tensor-parallel-size 8 --device xpu --dtype float16 --trust-remote-code
```

## Serve Large Models 

### Using Multiple Nodes

The following example serves `meta-llama/Llama-3.1-405B-Instruct` model using 2 nodes with `TP=8` and `PP=2`. Models exceeding 70 billion parameters generally require more than one Aurora node. First, use [`setup_ray_cluster.sh`] script to setup a Ray cluster across nodes. In `setup_ray_cluster.sh`, change `/path/to/setup_ray_cluster.sh` to a path in your environment. 

From a ray head node, initiate the Ray cluster and execute vLLM serve:
```bash linenums="1"
source /path/to/setup_ray_cluster.sh
main 
vllm serve meta-llama/Llama-3.1-405B-Instruct --port 8000 --tensor-parallel-size 8 --pipeline-parallel-size 2 --device xpu --dtype float16 --trust-remote-code --max-model-len 1024
```
Setting `--max-model-len` is important in order to fit this model on 2 nodes. In order to use higher `--max-model-len` values, you will need to use additonal nodes. 


## Examples 

### OpenAI API Example 

```bash
vllm serve meta-llama/Llama-2-7b-chat-hf --port 8000 --device xpu --dtype float16
python openai-example.py
```


### Offline Inference Benchmakr 

```bash
export VLLM_HOST_IP=$(getent hosts $(hostname).hsn.cm.aurora.alcf.anl.gov | awk '{ print $1 }' | tr ' ' '\n' | sort | head -n 1)
ray --logging-level debug start --head --verbose --node-ip-address=$VLLM_HOST_IP --port=6379 --num-cpus=64 --num-gpus=8&

python offline-benchmark.py
```