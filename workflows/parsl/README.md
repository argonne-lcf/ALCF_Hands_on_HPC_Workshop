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
import parsl
from parsl import python_app

# Scripts adapted from Parsl docs
# https://parsl.readthedocs.io/en/stable/1-parsl-introduction.html


# App that generates a random number after a delay
@python_app
def generate(limit, delay):
    from random import randint
    import time
    time.sleep(delay)
    return randint(1, limit)


with parsl.load():
    # Generate 5 random numbers between 1 and 10
    rand_nums = []
    for i in range(5):
        rand_nums.append(generate(10, i))

    # Wait for all apps to finish and collect the results
    outputs = [i.result() for i in rand_nums]

    # Print results
    print(outputs)
```

## Running tasks with sequential dependencies (2_sequential_workflow.py)

To create dependencies between tasks, make the creation of one future dependent on the result of another future.

```python
import parsl
from parsl import python_app, bash_app
from parsl.data_provider.files import File

# Scripts adapted from Parsl docs
# https://parsl.readthedocs.io/en/stable/1-parsl-introduction.html


# App that generates a random number
@python_app
def generate(limit):
    from random import randint
    return randint(1, limit)


# App that writes a variable to a file
@bash_app
def save(variable, outputs=[]):
    return 'echo %s &> %s' % (variable, outputs[0])


with parsl.load():
    # Generate a random number between 1 and 10
    random = generate(10)

    # This call will make the script wait before continuing
    print(f"Random number: {random.result()}")

    # Now, random has returned save the random number to a file
    saved = save(random, outputs=[File("sequential-output.txt")])

# Print the output file
with open(saved.outputs[0].result(), 'r') as f:
    print('File contents: %s' % f.read())
```

## Creating tasks within tasks (3_dynamic_workflow.py)

Sometimes it is advantageous for an app to call another app.  Apps that call other apps have a special type called a `join_app` (the join the results of other apps).  Here is an example:

```python
import parsl
from parsl.app.app import join_app, python_app

# Scripts adapted from Parsl docs
# https://parsl.readthedocs.io/en/stable/1-parsl-introduction.html


@python_app
def add(*args):
    """Add all of the arguments together. If no arguments, then
    zero is returned (the neutral element of +)
    """
    accumulator = 0
    for v in args:
        accumulator += v
    return accumulator


@join_app
def fibonacci(n):
    if n == 0:
        return add()
    elif n == 1:
        return add(1)
    else:
        return add(fibonacci(n - 1), fibonacci(n - 2))


with parsl.load():
    fib_series = fibonacci(10)

    print(fib_series.result())
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
import os
from parsl.config import Config

# PBSPro is the right provider for polaris:
from parsl.providers import PBSProProvider
# The high throughput executor is for scaling to HPC systems:
from parsl.executors import HighThroughputExecutor
# Use the MPI launcher
from parsl.launchers import MpiExecLauncher

# These options will run work in 1 node batch jobs run one at a time
nodes_per_job = 1
max_num_jobs = 1

# The config will launch workers from this directory
execute_dir = os.getcwd()

polaris_config = Config(
    executors=[
        HighThroughputExecutor(
            # Ensures one worker per GPU
            available_accelerators=4,
            max_workers_per_node=4,
            # Distributes threads to workers/GPUs in a way optimized for Polaris 
            cpu_affinity="list:24-31,56-63:16-23,48-55:8-15,40-47:0-7,32-39",
            # Increase if you have many more tasks than workers
            prefetch_capacity=0,
            # Needed to avoid interactions between MPI and os.fork
            provider=PBSProProvider(
                # Project name
                account="alcf_training",
                # Submission queue
                queue="HandsOnHPC",
                # Commands run before workers launched
                # Make sure to activate your environment where Parsl is installed
                worker_init=f'''source /grand/alcf_training/workflows_2024/_env/bin/activate;
                            cd {execute_dir}''',
                # Wall time for batch jobs
                walltime="0:05:00",
                # Change if data/modules located on other filesystem
                scheduler_options="#PBS -l filesystems=home:eagle:grand",
                # Ensures 1 manger per node and allows it to divide work to all 64 threads
                launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--ppn 1"),
                # options added to #PBS -l select aside from ncpus
                select_options="ngpus=4",
                # Number of nodes per batch job
                nodes_per_block=nodes_per_job,
                # Minimum number of batch jobs running workflow
                min_blocks=0,
                # Maximum number of batch jobs running workflow
                max_blocks=max_num_jobs,
                # Threads per node
                cpus_per_node=64,
            ),
        ),
    ],
    # How many times to retry failed tasks
    # this is necessary if you have tasks that are interrupted by a batch job ending
    retries=0,
)
```

## Example: Run hello_affinity on Polaris compute nodes (4_hello_polaris.py)

This script runs the application hello_affinity from our [GettingStarted](https://github.com/argonne-lcf/GettingStarted/tree/master/Examples/Polaris/affinity_gpu) repo.  It reports GPU and CPU affinities.  This script will run 4 instances of hello_affinity, one on each GPU of a polaris compute node in parallel.  It will create a batch job and block until the task futures are fulfilled by workers on the compute node.

```python
import parsl
import os
from parsl import bash_app
from config import polaris_config

# We will save outputs in the current working directory
working_directory = os.getcwd()

# Application that reports which worker affinities
@bash_app
def hello_affinity(stdout='hello.stdout', stderr='hello.stderr'):
    return '/grand/alcf_training/workflows_2024/GettingStarted/Examples/Polaris/affinity_gpu/hello_affinity'

# Load config for polaris
with parsl.load(polaris_config):

    # Create futures calling 'hello_affinity', store them in list 'tasks'
    tasks = []
    for i in range(4):
        tasks.append(hello_affinity(stdout=f"{working_directory}/output/hello_{i}.stdout",
                                    stderr=f"{working_directory}/output/hello_{i}.stderr"))

    # Wait on futures to return, and print results
    for i, t in enumerate(tasks):
        t.result()
        with open(f"{working_directory}/output/hello_{i}.stdout", "r") as f:
            print(f"Stdout of task {i}:")
            print(f.read())

    # Workflow complete!
    print("Hello tasks completed")
```

## Example: Run hello_affinity as an MPI application (5_mpi_app_example.py)

In the previous example, `mpiexec` was used as a launcher, rather than an executor.  In order to run applications that have MPI communication, `mpiexec` has to be used a different way by Parsl.  To run MPI applications, use the `SimpleLauncher` and the `MPIExecutor`.  Note that the configuration has to set `max_workers_per_block` to align with the resource needs of the application.  To run applications with different node numbers, a different `Config` object is needed.

```python
import parsl
import os
from parsl.config import Config
from parsl import bash_app
# PBSPro is the right provider for polaris:
from parsl.providers import PBSProProvider
# The MPIExecutor is for running MPI applications:
from parsl.executors import MPIExecutor
# Use the Simple launcher
from parsl.launchers import SimpleLauncher

# We will save outputs in the current working directory
working_directory = os.getcwd()

config = Config(
    executors=[
        MPIExecutor(
            max_workers_per_block=2,  # Assuming 2 nodes per task
            provider=PBSProProvider(
                account="alcf_training",
                worker_init=f"""source /grand/alcf_training/workflows_2024/_env/bin/activate; \
                                cd {working_directory}""",
                walltime="1:00:00",
                queue="HandsOnHPCScale",
                scheduler_options="#PBS -l filesystems=home:eagle:grand",
                launcher=SimpleLauncher(),
                select_options="ngpus=4",
                nodes_per_block=4,
                max_blocks=1,
                cpus_per_node=64,
            ),
        ),
    ]
)

resource_specification = {
  'num_nodes': 2,        # Number of nodes required for the application instance
  'ranks_per_node': 4,   # Number of ranks / application elements to be launched per node
  'num_ranks': 8,        # Number of ranks in total
}

@bash_app
def mpi_hello_affinity(parsl_resource_specification, depth=8,
                        stdout='mpi_hello.stdout', stderr='mpi_hello.stderr'):
    # PARSL_MPI_PREFIX will resolve to `mpiexec -n 8 -ppn 4 -hosts NODE001,NODE002`
    APP_DIR = "/grand/alcf_training/workflows_2024/GettingStarted/Examples/Polaris/affinity_gpu"
    return f"$PARSL_MPI_PREFIX --cpu-bind depth --depth={depth} \
            {APP_DIR}/set_affinity_gpu_polaris.sh {APP_DIR}/hello_affinity"

with parsl.load(config):
    tasks = []
    for i in range(4):
        tasks.append(mpi_hello_affinity(parsl_resource_specification=resource_specification,
                                        stdout=f"{working_directory}/mpi_output/hello_{i}.stdout",
                                        stderr=f"{working_directory}/mpi_output/hello_{i}.stderr"))
        
    # Wait on futures to return, and print results
    for i, t in enumerate(tasks):
        t.result()
        with open(f"{working_directory}/mpi_output/hello_{i}.stdout", "r") as f:
            print(f"Stdout of task {i}:")
            print(f.read())
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