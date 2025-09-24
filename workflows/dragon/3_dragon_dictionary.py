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

    # Destroy dictionary and free resources
    dist_dict.destroy()
