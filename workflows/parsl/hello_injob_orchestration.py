import parsl
import os
from parsl import python_app
from polaris_injob_config import polaris_config as config
# To run on Aurora, uncomment the following line and comment out the above line
# from aurora_injob_config import aurora_config as config

# We will save outputs in the current working directory
working_directory = os.getcwd()

# python app that reports worker affinities
@python_app
def hello_affinity():
    import os
    import socket
    import time

    time.sleep(1)  # Simulate some work being done

    hostname = socket.gethostname()

    # First look for cuda device
    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES")
    # If no cuda device set, look for intel device
    if gpu_id is None:
        gpu_id = os.environ.get("ZE_AFFINITY_MASK", "No GPUs assigned")

    return f"Hello from host {hostname}, GPU ID(s): {gpu_id}"

# Load config for polaris
with parsl.load(config):

    # Create futures calling 'hello_affinity', store them in list 'tasks'
    tasks = []
    for i in range(20):
        tasks.append(hello_affinity())
        
    # Wait on futures to return, and print results
    for i, t in enumerate(tasks):
        print(f"Result of task {i}: {t.result()}")

    # Workflow complete!
    print("Hello tasks completed")
