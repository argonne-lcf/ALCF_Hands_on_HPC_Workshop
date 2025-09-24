from globus_compute_sdk import Executor

# This script is intended to be run from your remote machine

# Scripts adapted from Globus Compute docs
# https://globus-compute.readthedocs.io/en/latest/quickstart.html

# First, define the function ...
def add_func(a, b):
    return a + b

# Paste your endpoint id here, e.g.
endpoint_id = ''

# Pre-staged endpoint for the live demo as a backup
# endpoint_id = '899ce0d3-c81f-4d56-b84d-8589f6b200a7'

# ... then create the executor, ...
with Executor(endpoint_id=endpoint_id) as gce:
    # ... then submit for execution, ...
    future = gce.submit(add_func, 5, 10)

    print("Submitted task to remote endpoint, waiting for result...")

    # ... and finally, wait for the result
    print(f"Remote result returned: add_func result={future.result()}")
