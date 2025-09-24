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

