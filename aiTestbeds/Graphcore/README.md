# Graphcore 

## Connection to Graphcore 

![Graphcore connection diagram](./graphcore_login.png)

Login to the Graphcore login node from your local machine.
Once you are on the login node, ssh to one of the Graphcore nodes.

```bash
ssh ALCFUserID@gc-login-01.ai.alcf.anl.gov
# or
ssh ALCFUserID@gc-login-02.ai.alcf.anl.gov

ssh gc-poplar-02.ai.alcf.anl.gov
# or
ssh gc-poplar-03.ai.alcf.anl.gov
# or
ssh gc-poplar-04.ai.alcf.anl.gov
```

## Clone Graphcore Examples

We use examples from [Graphcore Examples repository](https://github.com/graphcore/examples) for this hands-on. 
Clone the Graphcore Examples repository.
```bash
mkdir ~/graphcore
cd ~/graphcore
git clone https://github.com/graphcore/examples.git
cd examples
```

## Job Queuing and Submission

ALCF's Graphcore POD64 system uses Slurm for job submission and queueing. Below are some of the important commands for using Slurm.

* The Slurm command `srun` can be used to run individual Python scripts. Use the --ipus= option to specify the number of IPUs required for the run.
`srun --ipus=1 python mnist_poptorch.py`
* The jobs can be submitted to the Slurm workload manager through a batch script by using the `sbatch` command
* The `squeue` command provides information about jobs located in the Slurm scheduling queue.
* `SCancel` is used to signal or cancel jobs, job arrays, or job steps.

## Profiling 

We will use Pop Vision Graph Analyzer and System Analyzer to produce profiles. 

* [PopVision Graph Analyzer User Guide](https://docs.graphcore.ai/projects/graph-analyser-userguide/en/latest/)
* [PopVision System Analyzer User Guide](https://docs.graphcore.ai/projects/system-analyser-userguide/en/latest/)
* [PopVision Tools Downloads](https://www.graphcore.ai/developer/popvision-tools#downloads) 

#### PopVision Graph Analyzer

To generate a profile for PopVision Graph Analyzer, run the executable with the following prefix

```bash
$ POPLAR_ENGINE_OPTIONS='{"autoReport.all":"true", "autoReport.directory":"./graph_profile", "profiler.includeFlopEstimates": "true"}' python mnist_poptorch.py
```

This will generate all the graph profiling reports along with flops estimates and save the output to the graph_profile directory.

To visualize the profiles, download generated profiles to a local machine and open them using PopVision Graph Analyzer. 

#### PopVision System Analyzer

To generate a profile for PopVision System Analyzer, run the executable with the following prefix

```bash
$ PVTI_OPTIONS='{"enable":"true", "directory": "./system_profile"}' python mnist_poptorch.py
```
This will generate all the system profiling reports and save the output to system_profile directory.

To visualize the profiles, download generated profiles to a local machine and open them using PopVision Graph Analyzer. 


## Software Stack

The Graphcore Hands-on section consists of examples using  PyTorch and Poplar Software Stack. 

* [PyTorch](./PyTorch/)
* [Poplar](./Poplar/)

## Useful Resources 

* [Graphcore Documentation](https://docs.graphcore.ai/en/latest/)
* [Graphcore Examples Repository](https://github.com/graphcore/examples)
* Graphcore SDK Path: `/software/graphcore/poplar_sdk`
