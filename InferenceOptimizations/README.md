# LLM Inference Optimizations


* **Time:** October 30, 2024. 
  * 11.30 - 12.30 : LLM Inference Optimizations
* **Location:** TCS 1416 
* **Slack Channel:** [track-2](https://join.slack.com/share/enQtNzk3NzA1ODM1NjQwMC0yMTE4OTk3MGIwMmNiNzExZjMzZmY3YjMxNzRiNTI1MGEwNGZhMWEwMmM1NDhkOGZlZTY0NWE3ZTIzY2Q2NjE4) : Use to post questions, exact error messages etc. 

## Setup on Polaris 

* Get interactive node
  ```bash
  qsub -I -l select=1:ngpus=4 -l filesystems=home:eagle:grand -l walltime=1:00:00 -l -q HandsOnHPC -A alcf_training
  ```

* Create a new virtual environemnt for vLLM
    ```bash
    module use /soft/modulefiles/
    module load conda

    conda create -n vLLM_workshop python=3.11 -y
    ```
* Clone repo and install required dependancies 
  ```bash
  git clone https://github.com/argonne-lcf/ALCF_Hands_on_HPC_Workshop.git
  cd ALCF_Hands_on_HPC_Workshop/InferenceOptimizations

  pip install -r requirements.txt
  ```


## Hands-On Examples 

We will use LLAMA3-8B model to run inference hands-on examples. 

1. Inference with Huggingface
   ```bash
   $ bash run_HF.sh
   ```
   This script will run `run_HF.py` script with correct command line flags.  
   
2. Inference with vLLM
   ```bash
   $ bash run_vllm.sh
   ```
   This script will run `run_vllm.py` script with correct command line flags.

3. vLLM Quantization Example
   ```bash
   $ bash run_vllm_quant.sh
   ```
   This script will run `run_vllm.py` script with correct command line flags.

4. vLLM SD Example
   ```bash
   $ bash run_vllm_SD.sh
   ```
   This script will run `run_vllm.py` script with correct command line flags.



## Useful Links 

+ [Link to Presentation Slides](./HPC%20Workshop%20Inference%20Optimizations.pdf) 
+ [ALCF Hands-on  HPC Workshop Agenda](https://www.alcf.anl.gov/events/2024-alcf-hands-hpc-workshop)
+ [vLLM Documentation](https://docs.vllm.ai/en/latest/)
+ [vLLM Repo](https://github.com/vllm-project/vllm)
<!-- + [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/) -->
<!-- + [TensorRT-LLM Repo](https://github.com/NVIDIA/TensorRT-LLM/tree/main) -->

##### Acknowledgements

Contributors: [Krishna Teja Chitty-Venkata](https://krishnateja95.github.io/) and [Siddhisanket (Sid) Raskar](https://sraskar.github.io/). 

> This research used resources of the Argonne Leadership Computing Facility, which is a DOE Office of Science User Facility supported under Contract DE-AC02-06CH11357.