from globus_compute_sdk import Client

# This script is intended to be run from your remote machine

# Define a function that calls executable on Polaris
def hello_affinity(run_directory):
    import subprocess
    import os

    # This will create a run directory for the application to execute
    os.makedirs(os.path.expandvars(run_directory), exist_ok=True)
    os.chdir(os.path.expandvars(run_directory))

    # This is the command that calls the compiled executable
    command = "/grand/alcf_training/workflows/GettingStarted/Examples/Polaris/affinity_gpu/hello_affinity"

    # This runs the application command
    res = subprocess.run(command.split(" "), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Write stdout and stderr to files on Polaris filesystem
    with open("hello.stdout", "w") as f:
        f.write(res.stdout.decode("utf-8"))

    with open("hello.stderr", "w") as f:
        f.write(res.stderr.decode("utf-8"))

    # This does some error handling for safety, in case your application fails.
    # stdout and stderr are returned by the function
    if res.returncode != 0:
        raise Exception(f"Application failed with non-zero return code: {res.returncode} stdout='{res.stdout.decode('utf-8')}' stderr='{res.stderr.decode('utf-8')}'")
    else:
        return res.returncode, res.stdout.decode("utf-8"), res.stderr.decode("utf-8")


gcc = Client()
func_id = gcc.register_function(hello_affinity)
print(f"Registered hello_affinity; id {func_id}")
