# Integrating AI and Simulations

## Why Couple AI and Simulation?

### Active Learning
Or “experiment design”

Choose best simulations to launch 
- Diverse set?
- Ones that improve ML model? 
- Ones likely to improve quantity of interest? 
- If ML model degrading, collect more data? 

Examples:
- “Machine Learning Inter-Atomic Potentials Generation Driven by Active Learning” by Sivaraman, et al. [paper](https://arxiv.org/abs/1910.10254), [code](https://github.com/argonne-lcf/active-learning-md)
- "Coarse-Grained Density Functional Theory Predictions via Deep Kernel Learning" by Jackson and Sivaraman. [paper](https://chemrxiv.org/engage/chemrxiv/article-details/615c76ae7d3da5aa28edb9fb), [code](https://github.com/TheJacksonLab/ECG_ActiveLearning)


### Improve sampling of space
Examples:
- “Deep Learning Hamiltonian Monte Carlo” by Foreman, et al.  [paper](https://arxiv.org/pdf/2105.03418.pdf), [code](https://github.com/saforem2/l2hmc-qcd)
- “Stream-AI-MD: Streaming AI-driven Adaptive Molecular Simulations for Heterogeneous Computing Platform” by Brace, et al. [paper](https://dl.acm.org/doi/10.1145/3468267.3470578), [code](https://github.com/braceal/SC20-GB-CS1-ThetaGPU-AI-driven-MD) 


### Surrogate modeling
Replace part of simulation with ML surrogate model
- Expensive part?
- Inaccurate part? 
ML model output fed back into rest of simulation

Example:
“A turbulent eddy-viscosity surrogate modeling framework for RANS simulations” By Maulik, et al. [paper](https://doi.org/10.1016/j.compfluid.2020.104777), 
[code](https://github.com/argonne-lcf/TensorFlowFoam)

### Reduce I/O
- Apply ML model to save compressed simulation results 
- Train online during simulation (skip I/O bottleneck)
- In situ analysis giving feedback on simulations before completed
  - Need to adjust something? 
  
Example:
“In Situ Compression Artifact Removal in Scientific Data Using DeepTransfer 
Learning and Experience Replay” by Madireddy, et al. [paper](https://doi.org/10.1088/2632-2153/abc326)

### Control Simulation with ML
- Select simulation parameters
- Select numerical scheme

Example:
“Distributed Deep Reinforcement Learning for Simulation Control”
By Pawar & Maulik [paper](https://arxiv.org/pdf/2009.10306.pdf), [code](https://github.com/Romit-Maulik/RLLib_Theta/)

### Other Use Cases
- Data assimilation
- Augmenting simulation with ML closure/discrepancy model 
- Solver as part of ML loss function

## Ways to Couple AI and Simulation


See: “A terminology for in situ visualization and analysis systems” by Childs et al. 2020 

### How Close is the Coupling?
Proximity
- On node
- Off node, same computing resource
- Distinct computing resource

Data Access
- Direct: share same logical memory space (may or may not require copy)
- Indirect: distinct logical memory
- Either way: need to synchronize data 


### Division of Execution
- Space Division (Different physical compute resources)
  - Can allocate appropriate resource to each
  - But need to keep both utilized & transfer data

- Time Division (Some compute resources alternate simulation vs. AI)
  - Less or no synchronization & data transfer
  - But always blocking one or the other

### Example Modes
![Example Modes](ExampleModes.png)


