from globus_compute_sdk import Executor

# Define a function that calls executable on Polaris
def hello_affinity(run_directory):
    import subprocess, os
    os.makedirs(run_directory, exist_ok=True)
    os.chdir(run_directory)
    command = f"/home/csimpson/polaris/GettingStarted/Examples/Polaris/affinity_gpu/hello_affinity"
    res = subprocess.run(command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if res.returncode != 0:
        raise Exception(f"Application failed with non-zero return code: {res.returncode} stdout='{res.stdout.decode('utf-8')}' stderr='{res.stderr.decode('utf-8')}'")
    else:
        return res.returncode, res.stdout.decode("utf-8"), res.stderr.decode("utf-8")

# Paste your endpoint id here
endpoint_id = 'c0396551-2870-45f2-a2aa-70991eb120a4'

# Create executor for your endpoint
with Executor(endpoint_id=endpoint_id) as gce:
    # Create tasks
    tasks = []
    for i in range(4):
        tasks.append(gce.submit(hello_affinity,'/home/csimpson/workshop'))

    # Wait for tasks to return
    for t in tasks:
        print(t.result())