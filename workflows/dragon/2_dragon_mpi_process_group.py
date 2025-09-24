import os
import dragon
from dragon.infrastructure.policy import Policy
from dragon.native.machine import System, Node
from dragon.native.process_group import ProcessGroup
from dragon.native.process import ProcessTemplate
from dragon.infrastructure.facts import PMIBackend

# Optimal CPU and GPU affinities for Aurora Nodes
# gpu_affinities = [[float(f'{gid}.{tid}')] for gid in range(6) for tid in range(2)]
# cpu_affinities = [list(range(c, c+8)) for c in range(1, 52-8, 8)] + [list(range(c, c+8)) for c in range(53, 104-8, 8)]

# Set the PMI backend and path to MPI executable for Aurora
# pmi_backend = PMIBackend.PMIX
# exe = "" # Path to MPI executable for Aurora, TBD

# Optimal CPU and GPU affinities for Polaris Nodes
gpu_affinities = [[3],[2],[1],[0]]
cpu_affinities = [list(range(c, c+8)) for c in range(0, 32, 8)]

# Set the PMI backend and path to MPI executable for Polaris
pmi_backend = PMIBackend.CRAY
exe = "/grand/alcf_training/workflows/GettingStarted/Examples/Polaris/affinity_gpu/hello_affinity"


if __name__ == '__main__':

    # Number of processes to run in ProcessGroup
    alloc = System()
    nodelist = alloc.nodes
    num_procs_per_node = len(gpu_affinities) # This is different for Polaris and Aurora
    num_nodes = int(alloc.nnodes)
    num_procs = num_procs_per_node * num_nodes

    # Distribute MPI tasks with Policy and ProcessGroup
    # This will launch processes across nodes with specific CPU and GPU affinities
    print("\nLaunching MPI application with a ProcessGroup...", flush=True)
    # Create a ProcessGroup
    pg = ProcessGroup(pmi=pmi_backend) 

    # Create a list of policies that define the cpu and gpu affinities for each process
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
                    template=ProcessTemplate(target=exe,
                                            args=(), # No args for MPI executable
                                            cwd=os.getcwd(),
                                            policy=ppol,))
    pg.init()
    pg.start()
    pg.join()
    pg.close()

    