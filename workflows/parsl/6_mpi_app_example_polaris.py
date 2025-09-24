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

# Set your queue, account and environment
queue = "alcf_training"
account = "alcf_training"
load_env = f"source /grand/alcf_training/workflows/_env/bin/activate"

config = Config(
    executors=[
        MPIExecutor(
            max_workers_per_block=2,  # Assuming 2 nodes per task
            provider=PBSProProvider(
                account=account,
                worker_init=f"""{load_env};
                                cd {working_directory}""",
                walltime="00:10:00",
                queue=queue,
                scheduler_options="#PBS -l filesystems=home:eagle:grand",
                launcher=SimpleLauncher(),
                select_options="ngpus=4",
                nodes_per_block=2,
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
def mpi_hello_affinity(parsl_resource_specification, depth=8, stdout='mpi_hello.stdout', stderr='mpi_hello.stderr'):
    # PARSL_MPI_PREFIX will resolve to `mpiexec -n 8 -ppn 4 -hosts NODE001,NODE002`
    APP_DIR = "/grand/alcf_training/workflows/GettingStarted/Examples/Polaris/affinity_gpu"
    # wrap application with set_affinity_gpu_polaris.sh to set GPU affinity; see GettingStarted/Examples/Polaris/affinity_gpu for details
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