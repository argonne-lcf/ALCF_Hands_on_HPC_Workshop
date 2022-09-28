# Combining Simulation and AI/ML with SmartSim
Examples created by Riccardo Balin and edited by Filippo Simini at ALCF.


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
  - During online inference, the simulation uses an ML model to replace expensive or inacurate components
  - There are two components: the simulation and the SmartSim Orchestrator
  - Simulation sends model inputs for inference to database, evaluates any model and any pre- and post-processing computations within database, retreives the predictions, and keeps going with rest of computations
  - Compatible TensorFlow, TensorFlow Lite, Torch, and ONNXRuntime backends for model evaluations
  - Supports both both CPU and GPU backends enabling model evaluation on GPU
  - Simulation and model evaluation are loosely coupled -- run on separate resources but inference blocks simulation progress

| ![worflows](figures/train_inf_workflows.png) |
| ---- |
| Figure 1. Online training and inference workflows with SmartSim. |

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

| ![clustered](figures/clustered_approach.png) |
| ---- |
| Figure 2. Online training and inference with the clustered approach. |

| ![clustered](figures/colocated_approach.png) |
| ---- |
| Figure 3. Online training and inference with the co-located approach. |

| ![clustered](figures/cl_vs_coDB_scaling.png) |
| ---- |
| Figure 4. Comparison of average data transfer cost from simulation ranks to database for the co-located approach, clustered approach with 1 database node, and clustered approach with 4 database nodes as the number of simulation nodes grow.  |




## Installing SmartSim on Polaris

A Conda environment with the SmartSim and SmartRedis modules installed has been made available for you to use on Polaris. 
The examples below make use of this environment. 
You can activate it by executing
```
module load conda/2022-...
conda activate /path/to/ssim_env
```

Please note that this environment does not contain all the modules available with the base env from the `conda/2022-09-08` module, however it contains many of the essential packages, such as PyTorch, TensorFlow, Horovod, and MPI4PY.
If you wish to expand upon this Conda env, feel free to clone it or build your own version of it following this [installation script](installation/install_ssimEnv_Polaris.sh) and executing it with the command
```
source install_ssimEnv_Polaris.sh /path/to/conda/env
```
It is recommended you build the Conda env inside a project space rather than your home space on ALCF systems because it will produce a lot of files and consume disk space.

If you wish to use SmartSim on other ALCF systems (Theta and ThetaGPU), you can find instructions [here](https://github.com/rickybalin/ALCF/tree/main/SmartSim).



## Online Training of Turbulence Closure Model

In this first hands-on example, we will perform online training of a NN model of a polynomial function $y = f(x) = x^2 + 3x + 1$ in the domain $x \in [0, 10)$.
This example is available with both a Python and Fortran data producer, and implemented with both the clustered and co-located approaches at this [link](Polaris/).
Today, we will go through the [clustered Fortran example](Polaris/Fortan/train_clDB/), but we encourage you to give all of them a try.

Here is some information about the example:
- A Python driver script is used to launch the components of the workflow using the SmartSim API
- First, we launch a clustered database, which will run on one entire node
- Second, we launch the data producer, which is a Fortran program that performs the role of the simulation
  - This will run in parallel using MPI on a separate Polaris node. It will use the CPU of the node
  - It will connect the SmartRedis client to the database, send some useful meta-data, and then iterate over a time step loop which generates and sends training data to the database until training is complete
- Lastly, we launch the distributed training program that train the NN model
  - This will run in parallel on another separate node
  - It will use PyTorch and Horovod to perform data-parallel distributed training on the GPU
  - The model is a simple fully connected network with 2 hidden layers of 20 neurons, ReLU activatio functions, 1 input, $x$, and 1 output, $y=f(x)$
  - Training will progress until a tolerance on the average loss is reached, at which point a JIT-traced checkpoint of the model is saved to the disk and the simulator is told to quit

To build the Fortran data producer, follow the instructions below: 
- Change directory to `Polaris/Fortran/train_clDB/src`
- Make sure working directory is clean, if not run `./clean.sh`
- Make sure the environment is already loaded, if not run `source env_Polaris.sh`
- Build the executable with `./doConfig.sh`, which will be called `dataLoaderFtn.exe`


## Online Inference of Turbulence Closure Model

For the second hands-on example, we will perform online inference with the model for the polynomial we just trained in the previous example.
Similarly to online training, this example is available with a Fortran and Python reproducer of a simulation and is implemented with both approaches [here](Polaris/).
Today, we will go through the [Fortran co-located example](Polaris/Fortran/inference_coDB/).

Here is some information about the example:
- A Python driver script is used to launch the components of the workflow using the SmartSim API
- The co-located database and simulation are launched simultaneously, each sharing CPU resources on the Polaris nodes. Inference on the GPU on Polaris is coming very soon!
- The simulation will connect the SmartRedis client to the on-node database, upload the NN model to the database, and then iterate over a time step loop which generates inference data, sends it to the database, evaluates the model, and finally retreives the predictions
- The predictions are saved to a file for plotting and comparison with the true polynomial values

To build the Fortran simulation code, follow the build instructions from the previous example.

 


