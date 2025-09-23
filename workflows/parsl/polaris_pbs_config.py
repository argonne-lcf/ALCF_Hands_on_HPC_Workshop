import os
from parsl.config import Config
from parsl.addresses import address_by_interface

# PBSPro is the right provider for polaris:
from parsl.providers import PBSProProvider
# The high throughput executor is for scaling to HPC systems:
from parsl.executors import HighThroughputExecutor
# Use the MPI launcher
from parsl.launchers import MpiExecLauncher

# Set your queue and account
queue = "alcf_training"
account = "alcf_training"

# Set how to load environment
load_env = f"source /grand/alcf_training/workflows/_env/bin/activate"

# These options will run work in 1 node batch jobs run one at a time
nodes_per_job = 1
max_num_jobs = 1

# The config will launch workers from this directory
execute_dir = os.getcwd()

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
            # Use PBSPro as the job scheduler
            provider=PBSProProvider(
                # Project name
                account=account,
                # Submission queue
                queue=queue,
                # Commands run before workers launched
                # Make sure to activate your environment where Parsl is installed
                worker_init=f'''{load_env};
                            export TMPDIR=/tmp;
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
    retries=2,
)
