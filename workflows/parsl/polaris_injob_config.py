import os
from parsl.config import Config
from parsl.addresses import address_by_interface

# LocalProvider is for running orchestration with a job
from parsl.providers import LocalProvider
# The high throughput executor is for scaling to HPC systems:
from parsl.executors import HighThroughputExecutor
# Use the MPI launcher
from parsl.launchers import MpiExecLauncher


# Get the number of nodes:
node_file = os.getenv("PBS_NODEFILE")
with open(node_file,"r") as f:
    node_list = f.readlines()
    num_nodes = len(node_list)

polaris_config = Config(
    executors=[
        HighThroughputExecutor(
            # Specify network interface for the workers to connect to the Interchange
            address=address_by_interface('bond0'),
            # Ensures one worker per GPU
            available_accelerators=4,
            max_workers_per_node=4,
            # Distributes threads to workers/GPUs in a way optimized for Polaris 
            cpu_affinity="list:24-31,56-63:16-23,48-55:8-15,40-47:0-7,32-39",
            # Increase if you have many more tasks than workers
            prefetch_capacity=0,
            provider=LocalProvider(   
                # Distribute workers across all allocated nodes with mpiexec
                launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--ppn 1 --env TMPDIR=/tmp"),
                nodes_per_block=num_nodes,
                init_blocks=1,
                max_blocks=1,
            ),
        ),
    ],
    # How many times to retry failed tasks
    # this is necessary if you have tasks that are interrupted by a batch job ending
    retries=2,
)
