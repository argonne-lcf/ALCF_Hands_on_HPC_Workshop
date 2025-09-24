# Dragon HPC

Dragon HPC is a composable distributed run-time for managing processes, memory, and data at scale through high-performance communication objects.

This demo focuses on the Dragon's Python API (more information on Dragon's C API can be found in the [dragon docs](https://dragonhpc.org/portal/index.html)).

The Dragon python API uses python's `multiprocessing` API.  Dragon can therefore be used to extend scripts written for single shared memory devices with `multiprocessing` to run on multiple nodes on HPC systems.  Dragon also has a distributed dictionary that can be used by processes across the runtime to store and transfer data within memory pools arcross multiple nodes.  This demo will show multiple ways of running applications and tasks with Dragon process launching tools and how to make use of the distributed dictionary.

## Get Interactive Nodes

We'll run these tests in an interactive session, however a sample submit script is provided to test running in a batch job.

```shell
qsub -V -I -A alcf_training -l select=2 -l walltime=0:30:0 -l filesystems=home:eagle -q alcf_training
```

## Install

For the workshop, the prestaged environment with dragon can be loaded here:
```shell
module unload xalt
source /grand/alcf_training/workflows/_env/bin/activate
```

To create your own environment:
```shell
module unload xalt
module load conda
conda activate base
python -m venv _env
source _env/bin/activate
pip install dragonhpc
dragon-config add --ofi-runtime-lib=/opt/cray/libfabric/1.22.0/lib64
```

The last installation step of running `dragon-config` configures `dragon` to use fast RDMA transfers across Polaris' slingshot network.  Without this step, `dragon` would run in the default mode that uses slower TCP transfers.

## Dragon Pool

Dragon's Python API uses Python's `multiprocessing` API.  Dragon can therefore be used to extend scripts written for single shared memory devices with `multiprocessing` to run on multiple nodes.

This example shows how to run tasks with `Pool` in two different ways.  The first way uses `multiprocessing Pool` with the start method set to `dragon`.  This allows for pool processes to be distributed across nodes, but it is not possible to bind them to paricular gpus or cpus.

The second approach uses Dragon's native Pool object that allows a list of dragon Policies to be passed to the `Pool` that specify the gpu affinity.

These tests run a python function that sleeps and reports the hostname and GPU tile it sees pinned by `CUDA_VISIBLE_DEVICES` or `ZE_AFFINITY_MASK`.

Dragon scripts are launched with the `dragon` application, included in the demo environment.  To run the example script:

```shell
dragon 0_dragon_pool.py
```

**0_dragon_pool.py**
```python
import dragon
from dragon.native.machine import System
from multiprocessing import set_start_method, Pool
from dragon.native.pool import Pool as DragonPool
import numpy as np

# For Polaris, we have 4 GPUs per node
num_gpus_per_node = 4 # Assume one GPU/tile per process
# For Aurora, we have 12 GPU tiles per node
# num_gpus_per_node = 12 # Assume one GPU/tile per process

# A simple function to demonstrate task execution and GPU affinity
def hello_gpu_affinity(sleep_time):
    import os
    import socket
    import time

    time.sleep(sleep_time)  # Simulate some work being done
    hostname = socket.gethostname()

    # First look for cuda device
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES")
    # If no cuda device set, look for intel device
    if gpu_id is None:
        gpu_id = os.environ.get("ZE_AFFINITY_MASK", "No GPUs assigned")

    return f"Hello from host {hostname}, GPU ID(s): {gpu_id}"

if __name__ == '__main__':
    # Set the start method for multiprocessing to 'dragon'
    # This allows Dragon to manage process creation and affinity
    # This also allows for process launching across multiple nodes with the multiprocessing api
    set_start_method("dragon")

    # Set number of workers and tasks to run based on number of nodes
    alloc = System()
    num_nodes = int(alloc.nnodes)
    num_workers = num_gpus_per_node * num_nodes
    num_tasks = 2*num_workers
    # sleep_times are the inputs to the pool tasks
    sleep_times = np.ones(num_tasks) * 1.0  # Sleep for 1 second each

    # Test 1:
    # Distribute tasks across availble nodes with a simple pool
    # Unlike standard multiprocessing, Dragon will launch pool processes across multiple nodes
    # This pool does not use any GPU affinity
    print("Launching tasks with a simple Pool across nodes, no GPU affinity...", flush=True)
    pool = Pool(num_workers)
    async_results = pool.map_async(hello_gpu_affinity, sleep_times)
    results = async_results.get()
    for res in results:
        print(res, flush=True)
    pool.close()
    pool.join()

    # Test 2:
    # Distribute tasks across availble nodes with a Dragon Native Pool
    # Unlike a standard multiprocessing Pool, a Dragon Native Pool uses Dragon policies to launch processes
    # This pool binds 1 worker per GPU
    print("\nLaunching tasks with a Dragon Pool across nodes with GPU affinity...", flush=True)
    dragon_pool = DragonPool(policy=System().gpu_policies(), processes_per_policy=1)
    async_results = dragon_pool.map_async(hello_gpu_affinity, sleep_times)
    results = async_results.get()
    for res in results:
        print(res, flush=True)
    dragon_pool.close()
    dragon_pool.join()

```

## Dragon ProcessGroup

Dragon provides another way of distributing tasks to groups of workers called the dragon `ProcessGroup`.  Every process in the `ProcessGroup` can be given a specific node, gpu, and cpu, allowing for finegrained control of processes.

There are two example cases using `ProcessGroup`.  `1_dragon_process_group.py` distributes a simple python function to independent processes across the allocation.  This is done by assigning a `Policy` to each process in the group that specifies the node, gpu and cpu where it is to be run.

```shell
dragon 1_dragon_process_group.py
```

```python
import os
import dragon
from dragon.infrastructure.policy import Policy
from dragon.native.machine import System, Node
from dragon.native.process_group import ProcessGroup
from dragon.native.process import ProcessTemplate

# Optimal CPU and GPU affinities for Aurora Nodes
# gpu_affinities = [[float(f'{gid}.{tid}')] for gid in range(6) for tid in range(2)]
# cpu_affinities = [list(range(c, c+8)) for c in range(1, 52-8, 8)] + [list(range(c, c+8)) for c in range(53, 104-8, 8)]

# Optimal CPU and GPU affinities for Polaris Nodes
gpu_affinities = [[3],[2],[1],[0]]
cpu_affinities = [list(range(c, c+8)) for c in range(0, 32, 8)]


# A simple function to demonstrate task execution and GPU affinity
def hello_gpu_affinity(sleep_time):
    import os
    import socket
    import time
    
    time.sleep(sleep_time)  # Simulate some work being done

    hostname = socket.gethostname()

    # First look for cuda device
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES")
    # If no cuda device set, look for intel device
    if gpu_id is None:
        gpu_id = os.environ.get("ZE_AFFINITY_MASK", "No GPUs assigned")

    print(f"Hello from host {hostname}, GPU ID(s): {gpu_id}", flush=True)


if __name__ == '__main__':

    # Number of processes to run in ProcessGroup
    alloc = System()
    nodelist = alloc.nodes
    num_procs_per_node = len(gpu_affinities) # This is different for Polaris and Aurora
    num_nodes = int(alloc.nnodes)
    num_procs = num_procs_per_node * num_nodes

    # Test 1:
    # Distribute tasks with Policy and ProcessGroup
    # This will launch processes across nodes with specific CPU and GPU affinities
    print("\nLaunching tasks with simple GPU affinity policy with a ProcessGroup...", flush=True)
    # Create a ProcessGroup
    pg = ProcessGroup() 

    # Create a list of policies that set the cpu and gpu affinities for each process
    proc_policies = []
    for node in nodelist:
        node_name = Node(node).hostname
        for i in range(num_procs_per_node):
            ppol = Policy(host_name=node_name,
                        cpu_affinity=cpu_affinities[i],
                        gpu_affinity=gpu_affinities[i],
                        placement=Policy.Placement.HOST_NAME,)
            proc_policies.append(ppol)

    # Create a process for each policy in the ProcessGroup targeting the hello_gpu_affinity function
    for ppol in proc_policies:
        pg.add_process(nproc=1, 
                    template=ProcessTemplate(target=hello_gpu_affinity, # to run a compiled appication, set target to the path of compiled executable
                                                args=(1.0,), # sleep time
                                                cwd=os.getcwd(),
                                                policy=ppol,))
    pg.init()
    pg.start()
    pg.join()
    pg.close()

```

`2_dragon_mpi_process_group.py` uses `ProcessGroup` to run an MPI-enabled executable, where every process in the group is an MPI rank.  To enable message passing between processes in a process group, the `pmi` flag needs to be set in the `ProcessGroup` as shown in `2_dragon_mpi_process_group.py`.

## Dragon Distributed Dictionary

In addition to task launching, `dragon` provides a distributed data layer in its runtime called the `DDict` or Dragon Dictionary.  The `DDict` provisions pools of memory on nodes across the runtime where key-value data pairs can be stored.  Any process in the runtime can access any key-value pair in the `DDict`.  This is enabeld by dictionary manager processes that are created in the background within the runtime and manage the transfer of data between nodes.

`DDict`s can be used with `Pool` or `ProcessGroup`.  Multiple processes or `Pool`s and `ProcessGroup`s can use the same `DDict`.  This enables efficient sharing of data between processes in the runtime.

`3_dragon_dictionary.py` gives a simple example of how to create a `DDict` and use a `Pool` to store data within it.  The main process (running on the head node) then retrieves all the data stored in the `DDict`.

```shell
dragon 3_dragon_dictionary.py
```

```python
import dragon
from dragon.native.machine import System
from multiprocessing import Pool, set_start_method, current_process
from dragon.data.ddict import DDict

def setup(dist_dict):
    me = current_process()
    me.stash = {}
    me.stash["ddict"] = dist_dict

def assign(x):
    dist_dict = current_process().stash["ddict"]
    key = 'key_' + str(x)
    dist_dict[key] = x

if __name__ == '__main__':
    set_start_method("dragon")

    # Create a distributed dictionary with one manager per node across all allocated nodes
    alloc = System()
    num_nodes = int(alloc.nnodes)

    # Each node is allocated 1 GB of memory for the distributed dictionary
    # There is one dictionary manager per node
    print("Creating distributed dictionary with one manager per node...", flush=True)
    dist_dict = DDict(managers_per_node=1, n_nodes=num_nodes, total_mem=num_nodes*1024**3)

    # Use a multiprocessing Pool to assign values in parallel across nodes
    print("Assigning values to distributed dictionary from all nodes...", flush=True)
    with Pool(4*num_nodes, initializer=setup, initargs=(dist_dict,)) as pool:
        pool.map(assign, range(8*num_nodes))

    # Retrieve and print the contents of the distributed dictionary from all nodes
    print("Distributed dictionary contents:", flush=True)
    for k in dist_dict.keys():
        print(f"{k} = {dist_dict[k]}", flush=True)
```

## Submitting to PBS

To submit these tests to multiple nodes with PBS, use the following submit script:

```shell
qsub 4_submit_dragon.sh
```

**4_submit_dragon.sh**
```bash
#!/bin/bash -l
#PBS -A alcf_training
#PBS -l select=2
#PBS -N dragon_test
#PBS -l walltime=0:10:00
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -l place=scatter
#PBS -q alcf_training
#PBS -V

cd $PBS_O_WORKDIR

module unload xalt
source /grand/alcf_training/workflows/_env/bin/activate

dragon 0_dragon_pool.py
sleep 1
dragon 1_dragon_process_group.py
sleep 1
dragon 2_dragon_mpi_process_group.py
sleep 1
dragon 3_dragon_dictionary.py
```
