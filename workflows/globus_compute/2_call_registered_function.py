from globus_compute_sdk import Client
from globus_compute_sdk import Executor

# This script is intended to be run from your remote machine

# Paste your endpoint id here, e.g.
# endpoint_id = 'c0396551-2870-45f2-a2aa-70991eb120a4'
endpoint_id = ''

# Paste your hello_affinity function id here, e.g.
# affinity_func_id = '86afdb48-04e8-4e58-bfd1-cb2d8610a722'
affinity_func_id = ''

# Set a directory where application will run on Polaris file system
run_dir = '$HOME/workshop_globus_compute'

gce = Executor(endpoint_id=endpoint_id)

# Create tasks.  Task ids are saved t
task_ids = []
tasks = []
for i in range(4):
    tasks.append(gce.submit_to_registered_function(args=[f"{run_dir}/{i}"], function_id=affinity_func_id))

# Wait for tasks to return
for t in tasks:
    print(t.result())

print("Affinity Reported! \n")

# Print task execution details
gcc = Client()
for t in tasks:
    print(gcc.get_task(t.task_id),"\n")