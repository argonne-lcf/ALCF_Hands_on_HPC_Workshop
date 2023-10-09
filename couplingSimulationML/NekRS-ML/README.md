# Combining Simulation and AI/ML with SmartSim
Examples created by Riccardo Balin at ALCF.


## Introduction

SmartSim is an open source tool developed by the Hewlett Packard Enterprise (HPE) designed to facilitate the integration of traditional HPC simulation applications with machine learning workflows.
There are two core components to SmartSim:
- Infrastructure library (IL)
  - Provides API to start, stop and monitor HPC applications from Python
  - Interfaces with the scheduler launch jobs (PBSPro on Polaris and Cobalt on Theta/ThetaGPU)
  - Deploys a distributed in-memory database called the Orchestrator
- SmartRedis client library
  - Provides clients that connect to the Orchestrator from Fortran, C, C++, Python code
  - The client API library enables data transfer to/from database and ability to load and run JIT-traced Python and ML runtimes acting on stored data

For more resources on SmartSim, follow the links below:
- [Source code](https://github.com/CrayLabs/SmartSim)
- [Documentation](https://www.craylabs.org/docs/overview.html)
- [Zoo of examples](https://github.com/CrayLabs/SmartSim-Zoo)


## Deploying Workflows with SmartSim

There are two main types of workflows for combining simulation and ML in situ with SmartSim: online training and online inference.
- Online training
  - During online training, the simulation producing the training data and the ML training program using the data run simultaneously
  - There are three components: the data producer (e.g., a numerical simulation), the SmartSim Orchestrator, and the data consumer (e.g., a distributed training program)
  - Data flows from the simulation to the distributed training through the database
  - Training data is stored in-memory within the Orchestrator for the duration of the job, avoiding any I/O bottleneck and disk storage issues
  - Simulation and training are fully decoupled -- do not block each other and run on separate resources (CPU and/or GPU)
- Online inference
  - During online inference, the simulation uses an ML model to replace expensive or inaccurate computations
  - There are two components: the simulation and the SmartSim Orchestrator
  - Simulation sends model inputs for inference to database, evaluates any model and any pre- and post-processing computations within database, retreives the predictions, and keeps going with rest of computations
  - Compatible TensorFlow, TensorFlow Lite, Torch, and ONNXRuntime backends for model evaluations
  - Supports both both CPU and GPU backends enabling model evaluation on GPU
  - Simulation and model evaluation are loosely coupled -- run on separate resources but inference blocks simulation progress

| ![worflows](figures/smartsim_workflow_diagram.png) |
| ---- |
| Figure 1. Online training and inference workflows with SmartSim. Taken from [Balin et al. 2023](https://arxiv.org/abs/2306.12900v1). |

Additionally, there are two approaches to deploying the SmartSim workflow, both for training and inference: clustered and co-located.
- Clustered
  - SmartSim Orchestrator, simulation and ML component run on distinct set of nodes of the same machine
  - Deploy a single database sharded across a cluster of nodes
  - Pros: 
    - All training/inference data is contained in a single database and is visibible by any rank of simulation or ML applications
    - Offers the most flexibility to create complex workflows with additional components (e.g., add in situ visualization, train multiple models by connecting multiple ML applications to Orchestrator, run multiple simulations all contributing to training data set, and more ...)
  - Cons:
    - Reduced data transfer performance to/from Orchestrator as simulation and ML applications scale out
- Co-Located
  - SmartSim Orchestrator, simulation and ML component share resources on each node
  - Distinct database is deployed on each node
  - Pros:
     - Most efficient implementation to scale out (data transfer to/from database effectively constant with number of nodes!)
  - Cons:
    - Training/inference data is distributed across the various databases, accessing off-node data is non-trivial
    - This limits complexity of workflow and number of components deployed

| ![clustered](figures/colocated_vs_clustered.png) |
| ---- |
| Figure 2. Online training and inference with the colocated and clustered approaches on Polaris. Taken from [Balin et al. 2023](https://arxiv.org/abs/2306.12900v1). |


| ![clustered](figures/colocated_dataTransfer_weakScale.png) |
| ---- |
| Figure 3. Weak scaling of data send and retrieve operations between simulation reproducer and SmartSim database with the colocated deployment. Tests used up to 448 nodes of Polaris. Taken from [Balin et al. 2023](https://arxiv.org/abs/2306.12900v1). |

| ![clustered](figures/clustered_dataTransfer_weakScale.png) |
| ---- |
| Figure 4. Weak scaling of data send and retrieve operations between simulation reproducer and SmartSim database with the clustered deployment. Tests used up to 464 nodes of Polaris. Taken from [Balin et al. 2023](https://arxiv.org/abs/2306.12900v1). |

| ![clustered](figures/colocated_inference_scale.png) |
| ---- |
| Figure 4. Weak and strong scaling of ML model inference from simulation reproducer on SmartSim database with the clustered deployment. Tests used up to 448 nodes of Polaris and performed inference of the ResNet50 model. Taken from [Balin et al. 2023](https://arxiv.org/abs/2306.12900v1). |




## Installing SmartSim on Polaris

A Conda environment with the SmartSim and SmartRedis modules installed has been made available for you to use on Polaris. 
The examples below make use of this environment. 
You can activate it by executing
```
module load conda/2022-09-08
conda activate /eagle/projects/fallwkshp23/SmartSim/ssim
```

Please note that this environment does not contain all the modules available with the base env from the `conda/2022-09-08` module, however it contains many of the essential packages, such as PyTorch, TensorFlow, Horovod, and MPI4PY.
If you wish to expand upon this Conda env, feel free to clone it or build your own version of it following this [installation script](SmartSim_env/build_SSIM_Polaris.sh) and executing it *from a compute node* with the command
```
source build_SSIM_Polaris.sh
```
It is recommended you build the Conda env inside a project space rather than your home space on ALCF systems because it will produce a lot of files and consume disk space. 




## Online Training from NekRS of Wall Shear Stress Model


## Online Inference from NekRS of Wall Shear Stress Model




