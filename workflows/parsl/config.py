import os
from parsl.config import Config

# PBSPro is the right provider for polaris:
from parsl.providers import PBSProProvider
# The high throughput executor is for scaling to HPC systems:
from parsl.executors import HighThroughputExecutor
# Use the MPI launcher
from parsl.launchers import MpiExecLauncher

from parsl.addresses import address_by_interface

# These options will run work in 1 node batch jobs run one at a time
nodes_per_job = 1
max_num_jobs = 1

# The config will launch workers from this directory
execute_dir = os.getcwd()

polaris_config = Config(
    executors=[
        HighThroughputExecutor(
            # Ensures one worker per accelerator
            available_accelerators=4,
            address=address_by_interface('bond0'),
            # Distributes threads to workers sequentially in reverse order
            cpu_affinity="block-reverse",
            # Increase if you have many more tasks than workers
            prefetch_capacity=0,
            # Needed to avoid interactions between MPI and os.fork
            start_method="spawn",
            provider=PBSProProvider(
                # Project name
                account="datascience",
                # Submission queue
                queue="debug",
                # Commands run before workers launched
                worker_init=f'''source /eagle/fallwkshp23/workflows/env/bin/activate;
                            module load PrgEnv-nvhpc;
                            cd {execute_dir}''',
                # Wall time for batch jobs
                walltime="0:05:00",
                # Change if data/modules located on other filesystem
                scheduler_options="#PBS -l filesystems=home:eagle",
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
    # Retry failed tasks once
    retries=1,
)
