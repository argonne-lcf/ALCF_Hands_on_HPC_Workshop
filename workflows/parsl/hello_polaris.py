import parsl
import os
from parsl import bash_app
from config import polaris_config

# We will save outputs in the current working directory
working_directory = os.getcwd()

# Load config for polaris
parsl.load(polaris_config)


# Application that reports which GPU is assigned to the worker
@bash_app
def hello_device(stdout='hello.stdout', stderr='hello.stderr'):
    return 'echo Hello Polaris CUDA device $CUDA_VISIBLE_DEVICES on host $HOSTNAME'


# Application that reports which GPU is assigned to the worker
@bash_app
def hello_affinity(stdout='hello.stdout', stderr='hello.stderr'):
    # return 'echo Hello Polaris CUDA device $CUDA_VISIBLE_DEVICES on host $HOSTNAME'
    return '/home/csimpson/polaris/GettingStarted/Examples/Polaris/affinity_gpu/hello_affinity'


# Create futures calling 'hello_device', store them in list 'tasks'
tasks = []
for i in range(4):
    tasks.append(hello_affinity(stdout=f"{working_directory}/output/hello_{i}.stdout",
                                stderr=f"{working_directory}/output/hello_{i}.stderr"))

# Wait on futures to return, when they do, print stdout if successful, error otherwise
for i, t in enumerate(tasks):
    t.result()
    with open(f"{working_directory}/output/hello_{i}.stdout", "r") as f:
        print(f.read())

# Workflow complete!
print("Hello tasks completed")
