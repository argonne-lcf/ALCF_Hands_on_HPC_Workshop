import parsl
import os
from parsl import bash_app
from config import polaris_config

# We will save outputs in the current working directory
working_directory = os.getcwd()

# Load config for polaris
parsl.load(polaris_config)


# Application that reports which worker affinities
@bash_app
def hello_affinity(stdout='hello.stdout', stderr='hello.stderr'):
    return '/eagle/fallwkshp23/workflows/affinity_gpu/hello_affinity'


# Create futures calling 'hello_affinity', store them in list 'tasks'
tasks = []
for i in range(4):
    tasks.append(hello_affinity(stdout=f"{working_directory}/output/hello_{i}.stdout",
                                stderr=f"{working_directory}/output/hello_{i}.stderr"))

# Wait on futures to return, and print results
for i, t in enumerate(tasks):
    t.result()
    with open(f"{working_directory}/output/hello_{i}.stdout", "r") as f:
        print(f.read())

# Workflow complete!
print("Hello tasks completed")
