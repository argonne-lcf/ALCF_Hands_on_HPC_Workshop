# ALCF Hands-on HPC Workshop

## Virtual week Sep. 23 - 25, 2025
## In-person week October 7 - 9, 2025
## Agenda: https://www.alcf.anl.gov/events/2025-alcf-hands-hpc-workshop 

The ALCF is hosting a Hands-on HPC Workshop wihch features both a virtual event on September 23--25 and an in-person event on October 7--9, 2025, at the TCS Conference Center at Argonne National Laboratory.

The workshop will provide an opportunity for hands-on time on Aurora, Polaris and a few of the AI Testbeds focusing on porting applications to heterogeneous architectures (CPU + GPU), improving code performance, and exploring AI/ML applications development on ALCF systems.

<!--
The agenda below needs to be updated to reflect this years schedule
-->

## Overview 

The repo is divided into sections following the schedule of both the virtual and in-person weeks:

**Virtual Week, Day 1** 
  * Hardware Overview of Aurora
  * Building, Compiling and Running Applications on Aurora
  * CPU and GPU affinity on Aurora
  * [Programming Models](programmingModels)

**Virtual Week, Day 2**
   * [Intel's Data Parallel Extensions for Python (DPEP)](data_science/intel_python)
   * [Learning Frameworks & Distributed Deep Learning on Aurora](data_science/AI_frameworks)
   * [Foundation Models on Aurora](data_science/AI_frameworks/ezpz)
   * [Workflows](workflows)
   * [Inference Service](inference_service)
   * [Agentic Workflows](agentic_workflows)

**Virtual Week, Day 3**
   * [Tools](tools)


## Obtaining the material and running the examples

Please clone the repo while on a Polaris or Aurora login node:

```
git clone --recurse-submodules https://github.com/argonne-lcf/ALCF_Hands_on_HPC_Workshop.git
```

Then navigate to the particular material of interest and follow the instructions for executing the hands-on examples.



