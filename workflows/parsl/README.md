Parsl: Deploying Tasks with Parsl on ALCF Machines
===============================================

[Parsl](https://parsl.readthedocs.io/en/stable/) is a parallel programming library for Python.  It can be used to deploy large numbers of tasks with complex dependencies on ALCF machines, and is particularly well suited to run high-throughput workflows.  Parsl uses Python's concurrent futures module to create functions that return a Python futures object.  A Parsl workflow operates by creating futures for tasks that the Parsl executor will then fulfill by running them on available compute resources.

When a Parsl program runs and is configured to use Polaris compute resources, it will dynamically and elastically create batch jobs under the user's account on the Polaris scheduler.  These batch jobs will communicator with the Parsl process that launched then to acquire work and run it.

A Parsl workflow contains two parts:
* the workflow logic of functions and their dependencies
* the configuration of compute resources

We will begin by exploring how to define functions and dependencies.  Then we will describe how to configure resources to run the workflow on Polaris compute nodes.

# Setup and installation

First, login to Polaris and clone this repo:

```bash
# Login to Polaris
ssh polaris.alcf.anl.gov

# Clone the repo
git clone git@github.com:argonne-lcf/ALCF_Hands_on_HPC_Workshop.git
cd ALCF_Hands_on_HPC_Workshop/workflows/parsl
```

For the workshop, you can use the workshop python virtual environment that has parsl installed:
```bash
source /grand/alcf_training/workflows_2024/_env/bin/activate
```

To create your own environment:
```bash
module load conda
conda activate base
python -m venv _env
source _env/bin/activate
pip install parsl
```

# Parsl functions and logic

## Function app types (0_getting_started.py)

Parsl supports two main function types: the `python_app` type for running native python functions and the `bash_app` type that can be used to wrap around calls to a compiled executables.

This example demonstrates both app types:
```python
import parsl
from parsl import python_app, bash_app

# Scripts adapted from Parsl docs
# https://parsl.readthedocs.io/en/stable/1-parsl-introduction.html

# Python app type for running native python code
@python_app
def hello_python(message):
    return 'Hello %s' % message

# Bash app type for wrapping around calls to compiled code
@bash_app
def hello_bash(message, stdout='hello-stdout'):
    return 'echo "Hello %s"' % message

# This loads a default config that executes tasks on local threads
# To distribute to HPC resources on Polaris a different config needs
# to be loaded.  We'll cover this later.
with parsl.load():

    # invoke the Python app and print the result
    print(hello_python('World (Python)').result())

    # invoke the Bash app and read the result from a file
    hello_bash('World (Bash)').result()

with open('hello-stdout', 'r') as f:
    print(f.read())
```

## Running parallel tasks (1_parallel_workflow.py)

To run many tasks in parallel, create futures that call your app.  Once all futures have been created, only then wait on the results.

```python

```

## Running tasks with sequential dependencies (2_sequential_workflow.py)

To create dependencies between tasks, make the creation of one future dependent on the result of another future.

```python

```

## Creating tasks within tasks (3_dynamic_workflow.py)

Sometimes it is advantageous for an app to call another app.  Apps that call other apps have a special type called a `join_app` (the join the results of other apps).  Here is an example:

```python

```

# Parsl Configuration and Running on Polaris

The previous examples used Parsl's default configuration that runs tasks on local threads (in our case, threads on the polaris login node).

To run tasks on compute nodes we need to load a Polaris specific config object at the start of the Parsl workflow, e.g.:
```python
parsl.load(polaris_config)
```

Here we describe how to write a config for Polaris and demonstrate how to run tasks on compute nodes with it.

## Parsl Config for Polaris

The Parsl Config object describes how compute resources are assigned to Parsl workers.  Depending on how you want to use Polaris resources your Config may look different (e.g. if your tasks use gpus or only cpus).  Each Parsl worker will run one task at a time.  It contains many options, but the main aspects that need to be specified in the Config are:
* Executor: the executor describes how many workers will be available to the workflow and what Provider and Launcher will allocate and start workers.
* Provider: The provider describes how the Executor will get compute resources through the scheduler.
* Launcher: the Launcher describes how the Provider will place worker processes on compute resources, typically with an MPI in the HPC context.

On Polaris, for cases where you wish to run one task per gpu (a common use case), we recommend using the [`HighThroughputExecutor`](https://parsl.readthedocs.io/en/stable/stubs/parsl.executors.HighThroughputExecutor.html#parsl.executors.HighThroughputExecutor), the [`PbsProProvider`](https://parsl.readthedocs.io/en/stable/stubs/parsl.providers.PBSProProvider.html), and the [`MpiExecLauncher`](https://parsl.readthedocs.io/en/stable/stubs/parsl.launchers.MpiExecLauncher.html).

The Config object below will run 4 workers at a time.  These workers will be run on one Polaris node and each will access 1 GPU.

```python

```

## Example: Run hello_affinity on Polaris compute nodes (4_hello_polaris.py)

This script runs the application hello_affinity from our [GettingStarted](https://github.com/argonne-lcf/GettingStarted/tree/master/Examples/Polaris/affinity_gpu) repo.  It reports GPU and CPU affinities.  This script will run 4 instances of hello_affinity, one on each GPU of a polaris compute node in parallel.  It will create a batch job and block until the task futures are fulfilled by workers on the compute node.

```python

```


# A Note on Running a Parsl enabled script
As you have seen, a Parsl script will not return until all futures have been fulfilled.  The time this takes can depend on queue times and the overall runtime of the workflow.  Combined, this can be many hours.  It is therefore recommended to run Parsl scripts in a [screen](https://linuxize.com/post/how-to-use-linux-screen/) session or with [NoMachine](https://www.nomachine.com).

To run in screen:
```bash
$ screen -S parsl_session_name
$ python my_parsl_workflow.py

# the program begins execution, waiting for compute resources and for tasks to complete. While it runs, you can disconnect from the screen session with:
# Ctrl+a d
$
# Some time later, reconnect to your screen session from the same login node with 
$ screen -r parsl_session_name
$ # progress of your workflow script will then be displayed again
```