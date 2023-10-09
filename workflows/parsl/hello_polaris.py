import parsl
from parsl import bash_app
from configs import polaris_config

# Load config for polaris
parsl.load(polaris_config)

# Application that reports which GPU is assigned to the worker
@bash_app
def hello_device(stdout='hello.stdout', stderr='hello.stderr'):
    return 'echo "Hello Polaris CUDA device "$CUDA_VISIBLE_DEVICES" from Parsl"'

# Create futures calling 'hello_device', store them in list 'tasks'
tasks = []
for i in range(4):
    tasks.append(hello_device(stdout=f"./output/hello_{i}.stdout"))

# Wait on futures to return, when they do, print results
for i,t in enumerate(tasks):
    if t.result() == 0:
        with open(f"./output/hello_{i}.stdout", "r") as f:
            print(f.read())

# Workflow complete!
print("Hello tasks completed")