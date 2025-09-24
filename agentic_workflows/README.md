# Agentic Workflows at ALCF

## Overview

Agentic workflows are intelligent computational pipelines that use AI agents to coordinate and execute tasks, make real-time decisions based on user inputs or intermediate results, and adapt to changing conditions.
These workflows can:

* Reason about computational problems using large language models (LLMs)
* Respond to user inputs and prompts
* Dynamically adjust parameters based on intermediate results
* Orchestrate single or multi-system workflows across different computing resources
* Handle errors and retries intelligently
* Generate reports and insights from computational results
* Generate and test new code and agents on-the-fly to meet the workflow goals

This repository contains information on running agentic workflows at ALCF prepared for the 2025 Hands-On Workshop.
Please refer to the [slides](2025HandsOnWorkshop_agenticWorkflows.pdf) for more information on these workflows, how to run them on ALCF resources, and an example for how they can be useful to science applications.


## Examples of Agentic Workflows

Users can find examples on how to leverage ALCF systems to run agentic workflows at the [ALCF Agentic Workflows](https://github.com/argonne-lcf/alcf-agentics-workflow) public GitHub repository.
The repository currently contains two examples, one running the workflow in a remote fashion and the other with a local deployment.


### Multi-System (Remote) Workflow

In a multi-system or remote deployment of the workflow, the agents run and launch the computational components of the workflow in various ALCF or local systems.
In particular, the LLMs are served remotely, for example using the [ALCF Inference Endpoints](https://docs.alcf.anl.gov/services/inference-endpoints/).

Advantages of remote agentic workflows include:
* Lower implementation complexity
* Workflows can leverage ALCF services
* Users can target specific systems with each of the computational components, making efficient use of their allocation
* Better support for long-running workflows since the agents can run on local systems or login nodes

Disadvantages of remote agentic workflows include:
* Higher latency between tasks (including LLM inferencing)
* Tasks launching is subject to schedulers and queues

An example of a remote agentic workflow using the Crux and Polaris/Aurora systems at ALCF can be found [here](https://github.com/argonne-lcf/alcf-agentics-workflow/tree/main/remoteGlobusToAurora/TUTORIAL.md).

Notes for running the example during the ALCF Hands-On Workshop:
* When [generating the endpoint configuration](https://github.com/argonne-lcf/alcf-agentics-workflow/blob/main/remoteGlobusToAurora/TUTORIAL.md#13-generate-endpoint-configuration), make sure to specify the appropriate account and queue names. For this workshop, set both the queue and account names to `alcf_training`.
* In general, it is good practice to open the endpoint configuration file (`my-endpoint-config.yaml`) and ensure the parameters are correct, especially the paths included in the `worker_init` configuration since this code will be run at the start of the job on Aurora or Polaris.
* The Python major and minor versions (Python X.Y) should match across the two systems the workflow is run on. If not, the workflow will return a warning which could also lead to errors due to differences in serialization across versions.
* It is useful to run the workflow with the `--log-level DEBUG` flag to get more detailed logging during each of the steps (as indicated [here](https://github.com/argonne-lcf/alcf-agentics-workflow/blob/main/remoteGlobusToAurora/TUTORIAL.md#advanced-usage-examples)).


### Single-System (Local) Workflow

In a single-system or local deployment of the workflow, the agents and all computational components run on the same system, and even within the same batch job.
In particular, the LLMs are served locally on the compute nodes of the system, for example using [vLLM](https://docs.alcf.anl.gov/aurora/data-science/inference/vllm/).

Advantages of local agentic workflows include:
* Lower latency between tasks
* Running tasks bypassing the scheduler and system queues

Disadvantages of local agentic workflows include:
* Higher implementation complexity (need to serve LLMs locally)
* Subject to queue limitations (e.g., maximum run time)
* May consume allocation inefficiently 

An example of a local agentic workflow using the Polaris or Aurora systems at ALCF can be found [here](https://github.com/argonne-lcf/alcf-agentics-workflow/tree/main/localWorkflow).


Notes for running the example during the ALCF Hands-On Workshop:
* The model weights for the `Llama-2-7b-chat-hf` model have been made available in the workshop project space on the Eagle file system. Thus, the environment variables can be set up as follows
```
export HF_HOME=/eagle/alcf_training/model-weights/hub
export HF_DATASETS_CACHE=/eagle/alcf_training/model-weights/hub
export MODEL=meta-llama/Llama-2-7b-chat-hf
export TMPDIR="/tmp"
export HF_TOKEN="your_huggingface_token"
```
* When requesting an interactive node on Polaris or Aurora, make sure to use the appropriate queue and allocation names for the workshop. For this workshop, set both the queue and account names to `alcf_training`.
* It is useful to run the workflow with the `--log-level DEBUG` flag to get more detailed logging during each of the steps.

