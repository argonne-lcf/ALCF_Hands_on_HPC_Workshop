from globus_compute_sdk import Client
from globus_compute_sdk import Executor

# This script is intended to be run from your remote machine

# Paste your endpoint id here
endpoint_id = ''

# Paste your hello_affinity function id here
affinity_func_id = ''

# Prestaged endpoint and function as backup for live demo:
# endpoint_id = '899ce0d3-c81f-4d56-b84d-8589f6b200a7'
# affinity_func_id = '7755343d-b082-4a55-afcf-33f30d517bb9'

# Set a directory where application will run on Polaris file system
run_dir = '$HOME/workshop_globus_compute'

# Get an Executor client
gce = Executor(endpoint_id=endpoint_id)

# Create tasks.  Task ids are saved to list tasks
tasks = []
for i in range(32):
    tasks.append(gce.submit_to_registered_function(args=[f"{run_dir}/{i}"], function_id=affinity_func_id))

print("Submitted tasks to remote endpoint, waiting for results...")

# Wait for tasks to return
for t in tasks:
    print(t.result())

print("Affinity Reported! \n")

print("Details on task execution:")
# Print task execution details
# First get a client
gcc = Client()
# Loop through tasks and get task record
for t in tasks:
    print(gcc.get_task(t.task_id),"\n")
