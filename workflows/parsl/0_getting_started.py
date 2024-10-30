import parsl
from parsl import python_app, bash_app

# Scripts adapted from Parsl docs
# https://parsl.readthedocs.io/en/stable/1-parsl-introduction.html

# Python app type for running native python code
@python_app
def hello_python(message):
    return 'Hello %s' % message

# Bash app type for wrapping around calls to compiled code
@bash_app
def hello_bash(message, stdout='hello-stdout'):
    return 'echo "Hello %s"' % message

# This loads a default config that executes tasks on local threads
# To distribute to HPC resources on Polaris a different config needs
# to be loaded.  We'll cover this later.
with parsl.load():

    # invoke the Python app and print the result
    print(hello_python('World (Python)').result())

    # invoke the Bash app and read the result from a file
    hello_bash('World (Bash)').result()

with open('hello-stdout', 'r') as f:
    print(f.read())
