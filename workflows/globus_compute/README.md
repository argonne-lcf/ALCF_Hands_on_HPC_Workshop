Globus Compute: Remote execution of applications with Globus
===============================================

This tutorial demonstrates how to run applications on Polaris using [Globus Compute](https://www.globus.org/compute).  Globus Compute (formerly called FuncX) operates on a "fire-and-forget" model in which functions are sent to the Globus service to be deployed on a machine of the user's choice.  Globus Compute will communicate with a user process operating on a machine's login node callend an [endpoint](https://globus-compute.readthedocs.io/en/latest/endpoints.html).  The Globus Compute endpoint will use Parsl locally to communicate with the scheduler and execute work on compute nodes.

Globus Compute can be used to execute functions remotely as a service and can be integrated with [Globus Flows](https://docs.globus.org/api/flows/) to create workflows that automate the inegration of data transfers and function execution.

There are several requirements to deploying an application through Globus Compute on Polaris.
* An application that runs on Polaris compute nodes, either a compiled executable or a python function.
* A programming environment on Polaris with packages needed to run your your application and required globus packages installed.
* A programming environment on the machine from which you wish to deploy your functions with required globus packages installed.
* An active Globus Compute endpoint on a Polaris login node.
* An http connection on the remote machine deploying work.

# Setup

## Installing Globus modules

### Polaris

On Polaris, you will need to create a python virtual environment or a conda environment with the module `globus-compute-endpoint`.

For the workshop, you can use the workshop python virtual environment.  You will also need to unload xalt:
```bash
module unload xalt
source /grand/alcf_training/workflows/_env/bin/activate
```

To create your own environment:
```bash
module unload xalt
module load conda
conda activate base
python -m venv env
source env/bin/activate
pip install globus-compute-endpoint
```

Also, verify the version of python you are running on Polaris:
```bash
python --version
```

The pre-staged workshop environment has python version `3.11.8`, but make a note of the version in your environment if you have built your own, whichever you are running.

### Your remote machine

Your remote machine could be your laptop or a machine at another facility.  This remote machine will be the location from which you will send functions to the Globus Compute service.  You will need an environment with python 3.8+ to install the required globus compute client packages on the remote machine.  **It is recommended that you use the same major python version on the remote machine as you are using on Polaris.**  This may not be necessary for all functions, but the serialization and deserialization steps that Globus Compute will put your function through on the different machines may lead to incompatibility issues with different versions.  For this workshop, the functions we will test will be simple and this likely won't be an issue.

You will need to install the package `globus-compute-sdk` in this environment.  As an example, on a laptop running Mac OS with python installed through miniconda, a conda environment could be installed like this:
```bash
# Create remote machine conda environment with the same python version as the environment on Polaris
conda create -n workshop python==3.11.8
conda activate workshop

# Install globus compute client module
pip install globus-compute-sdk
```

## Creating and Starting an Endpoint on Polaris

Login to Polaris and clone this repo.  Activate your environment.
```bash
# Login to Polaris
ssh polaris.alcf.anl.gov

# Clone the repo
git clone git@github.com:argonne-lcf/ALCF_Hands_on_HPC_Workshop.git
cd ALCF_Hands_on_HPC_Workshop/workflows/globus_compute

# If you haven't already, activate the environment
source /grand/alcf_training/workflows_2024/_env/bin/activate
```

Use the sample config [polaris_config.yaml](polaris_config.yaml) provided to configure and start your endpoint.  The sample config has similar features to the Parsl config and looks like this:
```yaml
engine:
    # This engine uses the HighThroughputExecutor
    type: GlobusComputeEngine
    
    available_accelerators: 4 # Assign one worker per GPU
    max_workers_per_node: 4
    
    cpu_affinity: "list:24-31,56-63:16-23,48-55:8-15,40-47:0-7,32-39"
    
    prefetch_capacity: 0  # Increase if you have many more tasks than workers                                              
    max_retries_on_system_failure: 2

    strategy: simple
    job_status_kwargs:
        max_idletime: 300
        strategy_period: 60

    provider:
        type: PBSProProvider

        launcher:
            type: MpiExecLauncher
            # Ensures 1 manger per node, work on all 64 cores
            bind_cmd: --cpu-bind
            overrides: --ppn 1

        account: alcf_training
        queue: HandsOnHPC
        cpus_per_node: 64
        select_options: ngpus=4

        # e.g., "#PBS -l filesystems=home:grand:eagle\n#PBS -k doe"
        scheduler_options: "#PBS -l filesystems=home:eagle:grand"

        # Node setup: activate necessary conda environment and such
        worker_init: "source /grand/alcf_training/workflows_2024/_env/bin/activate; module load PrgEnv-nvhpc; cd $HOME/.globus_compute/workshop-endpoint"

        walltime: 00:30:00
        nodes_per_block: 1
        init_blocks: 0
        min_blocks: 0
        max_blocks: 1
```

The config `aurora_config.yaml` can be used in the same way to set up an endpoint on Aurora that will run on worker per GPU tile.

There will be a command line tool `globus-compute-endpoint` in your path that will allow you to manage your endpoint process.  

Congiure and start the endpoint:
```bash
globus-compute-endpoint configure --endpoint-config ./polaris_config.yaml workshop-endpoint
globus-compute-endpoint start workshop-endpoint
```
During this step, if this is your first time using globus tools, you will be prompted to validate your ALCF credentials with the Globus.  When this happens, a URL to the globus service will be given to you at the command line; paste this into a web brower and follow the instructions.  You will need your MobilePass+ credentials (that you use to login to ALCF machines) during this step.  Once the Globus website completes your credential validation, it will give you a code that you will paste back into your shell.  One this code is accepted, your credential validation for this endpoint will be complete and you can interact with the endpoint through the globus service.

Verify that your endpoint is active.
```bash
globus-compute-endpoint list
```

Your endpoint will have and id, copy this unique id, you will need it on your local machine.

If you need to make changes to your endpoint config, look for the file `$HOME/.globus_compute/workshop-endpoint/config.yaml` and make changes.  Then restart the endpoint process to activate the changes:
```bash
globus-compute-endpoint restart workshop-endpoint
```

You can also verify that your endpoint is communicating with the Globus Service by looking at https://app.globus.org/compute.

# Examples
## Remote execution of a simple function (0_remote_adder.py)

We can send a simple function, adding two numbers to your endpoint.  To run this script, paste your endpoint's id into the script below.  Like Parsl functions, a Globus Compute function returns a future.  Include a call that waits on the result of the future to make your script wait for the result to return.

```python
from globus_compute_sdk import Executor

# This script is intended to be run from your remote machine

# Scripts adapted from Globus Compute docs
# https://globus-compute.readthedocs.io/en/latest/quickstart.html

# First, define the function ...
def add_func(a, b):
    return a + b

# Paste your endpoint id here
endpoint_id = ''

# ... then create the executor, ...
with Executor(endpoint_id=endpoint_id) as gce:
    # ... then submit for execution, ...
    future = gce.submit(add_func, 5, 10)

    # ... and finally, wait for the result
    print(future.result())
```

## Register a function with Globus Service (1_register_function.py)

A function can be registered with the Globus service to be executed later.  It can be called with an id.

This function wraps around a compiled executable, in this example a program that reports cpu and gpu affinity.  This example both saves stdout and stderr to the file system on Polaris and returns it as the function result which can be accessed from the remote machine.

```python
import globus_compute_sdk

# This script is intended to be run from your remote machine

# Define a function that calls executable on Polaris
def hello_affinity(run_directory):
    import subprocess, os

    # This will create a run directory for the application to execute
    os.makedirs(os.path.expandvars(run_directory), exist_ok=True)
    os.chdir(os.path.expandvars(run_directory))

    # This is the command that calls the compiled executable
    command = f"/grand/alcf_training/workflows_2024/GettingStarted/Examples/Polaris/affinity_gpu/hello_affinity"

    # This runs the application command
    res = subprocess.run(command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Write stdout and stderr to files on Polaris filesystem
    with open("hello.stdout","w") as f:
        f.write(res.stdout.decode("utf-8"))
    
    with open("hello.stderr","w") as f:
        f.write(res.stderr.decode("utf-8"))

    # This does some error handling for safety, in case your application fails.
    # stdout and stderr are returned by the function
    if res.returncode != 0:
        raise Exception(f"Application failed with non-zero return code: {res.returncode} stdout='{res.stdout.decode('utf-8')}' stderr='{res.stderr.decode('utf-8')}'")
    else:
        return res.returncode, res.stdout.decode("utf-8"), res.stderr.decode("utf-8")
    
gc = globus_compute_sdk.Client()
fusion_func = gc.register_function(hello_affinity)
print(f"Registered hello_affinity; id {fusion_func}")
```

## Call registered function (2_call_registered_function.py)

This script shows how to call the function registered in the previous example.  Copy the function id printed from that example in the script below.  Also paste in your endpoint id.

```python
from globus_compute_sdk import Client
from globus_compute_sdk import Executor

# This script is intended to be run from your remote machine

# Paste your endpoint id here, e.g.
# endpoint_id = 'c0396551-2870-45f2-a2aa-70991eb120a4'
endpoint_id = ''

# Paste your hello_affinity function id here, e.g.
# affinity_func_id = '86afdb48-04e8-4e58-bfd1-cb2d8610a722'
affinity_func_id = ''

# Set a directory where application will run on Polaris file system
run_dir = '$HOME/workshop_globus_compute'

gce = Executor(endpoint_id=endpoint_id)

# Create tasks.  Task ids are saved t
task_ids = []
tasks = []
for i in range(4):
    tasks.append(gce.submit_to_registered_function(args=[f"{run_dir}/{i}"], function_id=affinity_func_id))

# Wait for tasks to return
for t in tasks:
    print(t.result())

print("Affinity Reported! \n")

# Print task execution details
gcc = Client()
for t in tasks:
    print(gcc.get_task(t.task_id),"\n")
```

# Next Steps

Globus compute allows for the remote execution of applications on ALCF machines.  Often, projects wish to couple the remote execution of tasks with data transfers or the execution of other associated tasks.  [Globus Flows](https://docs.globus.org/api/flows/) allows for the integration of Globus Compute tasks with other actions.
