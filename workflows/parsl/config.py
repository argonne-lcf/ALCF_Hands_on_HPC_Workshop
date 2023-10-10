from parsl.config import Config

# PBSPro is the right provider for polaris:
from parsl.providers import PBSProProvider
# The high throughput executor is for scaling to HPC systems:
from parsl.executors import HighThroughputExecutor
# Use the MPI launcher
from parsl.launchers import MpiExecLauncher

from parsl.addresses import address_by_interface


run_dir="/home/csimpson/polaris/demo_materials/parsl/"

num_nodes = 1
polaris_config = Config(
    executors=[
        HighThroughputExecutor(
            available_accelerators=4,  # Ensures one worker per accelerator
            address=address_by_interface('bond0'),
            cpu_affinity="block-reverse",  # Prevents thread contention
            prefetch_capacity=0,  # Increase if you have many more tasks than workers
            start_method="spawn",  # Needed to avoid interactions between MPI and os.fork
            retries=2, # Retry failed tasks up to two times
            provider=PBSProProvider(
                account="datascience", # Project name
                queue="debug", # Submission queue
                worker_init="source /home/csimpson/polaris/env_parsl/bin/activate ; cd "+run_dir, # worker initialization commands
                walltime="0:05:00", # Wall time for batch jobs
                scheduler_options="#PBS -l filesystems=home:eagle",  # Change if data/modules on other filesystem
                launcher=MpiExecLauncher(
                    bind_cmd="--cpu-bind", overrides="--ppn 1"
                ),  # Ensures 1 manger per node and allows it to divide work to all 64 threads
                select_options="ngpus=4", # options passed to #PBS -l select
                nodes_per_block=num_nodes, # Number of nodes per batch job
                min_blocks=0, # Minimum number of batch jobs running workflow
                max_blocks=1, # Maximum number of batch jobs running workflow
                cpus_per_node=64, # Threads per node
            ),
        ),
    ]
)
