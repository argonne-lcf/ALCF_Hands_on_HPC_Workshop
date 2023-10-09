import os
import parsl
from parsl import python_app, bash_app
from parsl.data_provider.files import File

# Scripts adapted from Parsl docs
# https://parsl.readthedocs.io/en/stable/1-parsl-introduction.html

# App that generates a random number
@python_app
def generate(limit):
      from random import randint
      return randint(1,limit)

# App that writes a variable to a file
@bash_app
def save(variable, outputs=[]):
      return 'echo %s &> %s' % (variable, outputs[0])

parsl.load()

# Generate a random number between 1 and 10
random = generate(10)

# This call will make the script wait before continuing
print('Random number: %s' % random.result())

# Now, random has returned save the random number to a file
saved = save(random, outputs=[File(os.path.join(os.getcwd(), 'sequential-output.txt'))])

# Print the output file
with open(saved.outputs[0].result(), 'r') as f:
      print('File contents: %s' % f.read())
