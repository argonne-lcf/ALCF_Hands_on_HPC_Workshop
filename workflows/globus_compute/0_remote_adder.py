from globus_compute_sdk import Executor

# This script is intended to be run from your remote machine

# Scripts adapted from Globus Compute docs
# https://globus-compute.readthedocs.io/en/latest/quickstart.html


# First, define the function ...
def add_func(a, b):
    return a + b


# Paste your endpoint id here, e.g.
# endpoint_id = 'c0396551-2870-45f2-a2aa-70991eb120a4'
endpoint_id = ''

# ... then create the executor, ...
with Executor(endpoint_id=endpoint_id) as gce:
    # ... then submit for execution, ...
    future = gce.submit(add_func, 5, 10)

    print("Submitted task to remote endpoint, waiting for result...")

    # ... and finally, wait for the result
    print(future.result())
