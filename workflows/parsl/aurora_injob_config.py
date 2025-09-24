import os
from parsl.config import Config
from parsl.addresses import address_by_interface

from parsl.providers import LocalProvider
# The high throughput executor is for scaling to HPC systems:
from parsl.executors import HighThroughputExecutor
# Use the MPI launcher
from parsl.launchers import MpiExecLauncher

# Set your queue and account
queue = "alcf_training"
account = "alcf_training"

# Set how to load environment
load_env = f"source /flare/alcf_training/workflows/_env/bin/activate"

# Get the number of nodes:
node_file = os.getenv("PBS_NODEFILE")
with open(node_file,"r") as f:
    node_list = f.readlines()
    num_nodes = len(node_list)

tile_names = [f'{gid}.{tid}' for gid in range(6) for tid in range(2)]

aurora_config = Config(
    executors=[
        HighThroughputExecutor(
            # Specify network interface to use to connect worker nodes to interchange
            address=address_by_interface('bond0'),
            # Ensures one worker per GPU
            # Since Aurora tile affinity can be a non-integer format, we use a list of strings here
            # e.g. ['0.0', '0.1', '1.0', '1.1', ..., '5.1']
            available_accelerators=tile_names,
            max_workers_per_node=12,
            # Distributes threads to workers/GPUs in a way optimized for Aurora
            cpu_affinity="list:1-8,105-112:9-16,113-120:17-24,121-128:25-32,129-136:33-40,137-144:41-48,145-152:53-60,157-164:61-68,165-172:69-76,173-180:77-84,181-188:85-92,189-196:93-100,197-204",
            # Increase if you have many more tasks than workers
            prefetch_capacity=0,
            # Use LocalProvider for in-job execution
            provider=LocalProvider(
                # Ensures 1 manger per node and allows it to divide work to all 64 threads
                launcher=MpiExecLauncher(bind_cmd="--cpu-bind", overrides="--ppn 1"),
                # Number of nodes per batch job
                nodes_per_block=num_nodes,
                # Maximum number of batch jobs running workflow
                max_blocks=1,
                init_blocks=1,
            ),
        ),
    ],
    # How many times to retry failed tasks
    # this is necessary if you have tasks that are interrupted by a batch job ending
    retries=2,
    # Turning off logging of the interchange can improve performance above 500 nodes
    initialize_logging=False,
)
