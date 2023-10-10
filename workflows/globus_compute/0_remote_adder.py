from globus_compute_sdk import Executor

# First, define the function ...
def add_func(a, b):
    return a + b

tutorial_endpoint_id = 'c0396551-2870-45f2-a2aa-70991eb120a4'
# ... then create the executor, ...
with Executor(endpoint_id=tutorial_endpoint_id) as gce:
    # ... then submit for execution, ...
    future = gce.submit(add_func, 5, 10)

    # ... and finally, wait for the result
    print(future.result())