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

    